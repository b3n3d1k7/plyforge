#!/usr/bin/env python3
"""
PlyForge
========

Generate 1:1 ply cutting templates for composite layups by non-planar
slicing of a mold or finished-part STL.

Layers are generated from the top AND bottom surfaces of the mesh,
working inwards toward the middle. Each layer outline is exported as a
PNG that can be printed at 100% and used directly as a ply template for
glass-fiber or carbon-fiber layup, or for a two-part mold of a hydrofoil
wing.

Features:
    - white background, black outline of the current layer
    - optional light-blue dashed outline of a reference layer
      (the outermost layer of the same side by default)
    - 5 mm minor grid, labeled ticks with configurable step
    - equal aspect ratio, axes in millimeters
    - both axes start at 0 (coordinates are shifted by the mesh minimum)
    - title showing "Top layer N" or "Bottom layer N"
    - 1:1 physical print scale by default; DPI controls pixel density only
"""

import argparse
import os
import numpy as np
import trimesh
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from skimage import measure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PROG = "plyforge"

FONT_SIZE = 7

MARGIN_LEFT_IN   = 1.6
MARGIN_RIGHT_IN  = 0.4
MARGIN_BOTTOM_IN = 1.4
MARGIN_TOP_IN    = 0.7

MM_PER_INCH = 25.4


# ----------------------------------------------------------------------
# PNG output
# ----------------------------------------------------------------------
def write_png(lines, out_path, bounds, title,
              reference=None, dpi=200, print_scale=True,
              label_step_x=5.0, label_step_y=5.0):
    """
    Render polylines to a PNG.

    Coordinates are shifted so that the mesh minimum becomes 0 on both
    axes. `label_step_x` / `label_step_y` control the labeled tick
    interval in mm; if either is <= 0, an auto step targeting ~20 labels
    is used instead.
    """
    xmin, xmax, ymin, ymax = bounds
    shift = np.array([xmin, ymin], dtype=float)

    def shift_lines(ls):
        return [l - shift for l in ls]

    lines = shift_lines(lines)
    if reference:
        reference = shift_lines(reference)

    xlim = (0.0, float(xmax - xmin))
    ylim = (0.0, float(ymax - ymin))

    plot_w_mm = xlim[1] - xlim[0]
    plot_h_mm = ylim[1] - ylim[0]

    if print_scale:
        plot_w_in = plot_w_mm / MM_PER_INCH
        plot_h_in = plot_h_mm / MM_PER_INCH

        fig_w = MARGIN_LEFT_IN + plot_w_in + MARGIN_RIGHT_IN
        fig_h = MARGIN_BOTTOM_IN + plot_h_in + MARGIN_TOP_IN

        if plot_w_in < 0.5 or plot_h_in < 0.5:
            print(f"  warning: plot area is very small "
                  f"({plot_w_mm:.1f} x {plot_h_mm:.1f} mm). "
                  "Consider using --no-print-scale.")
        if fig_w > 50.0 or fig_h > 50.0:
            print(f"  warning: figure is very large "
                  f"({fig_w:.1f} x {fig_h:.1f} in). "
                  "Consider using --no-print-scale.")

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_axes([
            MARGIN_LEFT_IN / fig_w,
            MARGIN_BOTTOM_IN / fig_h,
            plot_w_in / fig_w,
            plot_h_in / fig_h,
        ])
    else:
        fig_w = 10.0
        fig_h = fig_w * plot_h_mm / plot_w_mm
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def resolve_step(user_step, lo, hi, target=20):
        if user_step and user_step > 0:
            return float(user_step)
        span = hi - lo
        if span <= 0:
            return 5.0
        raw = span / target
        return max(5.0, np.ceil(raw / 5.0) * 5.0)

    step_x = resolve_step(label_step_x, xlim[0], xlim[1])
    step_y = resolve_step(label_step_y, ylim[0], ylim[1])

    # Minor grid every 5 mm; labeled ticks on the requested step
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(step_x))
    ax.yaxis.set_major_locator(MultipleLocator(step_y))

    ax.grid(True, which="minor", color="#e6e6e6",
            linewidth=0.4, linestyle=":")
    ax.grid(True, which="major", color="#b0b0b0",
            linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    if reference:
        for line in reference:
            if len(line) < 2:
                continue
            ax.plot(line[:, 0], line[:, 1],
                    color="#4a90e2", linewidth=0.7, linestyle="--",
                    solid_joinstyle="round", solid_capstyle="round",
                    zorder=1)

    for line in lines:
        if len(line) < 2:
            continue
        ax.plot(line[:, 0], line[:, 1],
                color="black", linewidth=0.8,
                solid_joinstyle="round", solid_capstyle="round",
                zorder=2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title(title, fontsize=FONT_SIZE, pad=4)
    ax.set_xlabel("X (mm)", fontsize=FONT_SIZE)
    ax.set_ylabel("Y (mm)", fontsize=FONT_SIZE)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=FONT_SIZE)
    plt.setp(ax.get_yticklabels(), fontsize=FONT_SIZE)
    ax.tick_params(axis="both", which="major", pad=2, labelsize=FONT_SIZE)

    if not print_scale:
        fig.tight_layout()

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Height-field extraction
# ----------------------------------------------------------------------
def build_height_fields(mesh, resolution):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xs, ys)

    origins = np.column_stack([
        X.ravel(), Y.ravel(), np.full(X.size, zmin - 1.0)
    ])
    dirs = np.tile([0.0, 0.0, 1.0], (X.size, 1))

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True
    )

    top = np.full(X.size, np.nan)
    bot = np.full(X.size, np.nan)

    order = np.argsort(index_ray)
    index_ray = index_ray[order]
    locations = locations[order]

    splits = np.searchsorted(index_ray, np.arange(X.size))
    splits = np.append(splits, len(index_ray))

    for i in range(X.size):
        z_hits = locations[splits[i]:splits[i + 1], 2]
        if z_hits.size > 0:
            top[i] = z_hits.max()
            bot[i] = z_hits.min()

    top = top.reshape(X.shape)
    bot = bot.reshape(X.shape)

    valid = ~(np.isnan(top) | np.isnan(bot))
    if not valid.any():
        raise SystemExit("No ray hits - mesh may be empty or inverted.")

    vp = np.column_stack([X[valid], Y[valid]])
    top_filled = griddata(vp, top[valid], (X, Y), method="nearest")
    bot_filled = griddata(vp, bot[valid], (X, Y), method="nearest")

    top_filled = gaussian_filter(top_filled, sigma=1.5)
    bot_filled = gaussian_filter(bot_filled, sigma=1.5)

    return X, Y, top_filled, bot_filled, valid


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def cos_slope(field, dx, dy):
    gy, gx = np.gradient(field, dy, dx)
    return 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)


def extract_contours(field, level, valid, X, Y):
    masked = np.where(valid, field, -1.0)
    masked = np.pad(masked, 1, mode="constant", constant_values=-1.0)

    contours = measure.find_contours(masked, level)

    xs = X[0, :]
    ys = Y[:, 0]
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    x0 = xs[0] - dx
    y0 = ys[0] - dy

    lines = []
    for c in contours:
        x_world = x0 + c[:, 1] * dx
        y_world = y0 + c[:, 0] * dy
        if len(x_world) >= 3:
            lines.append(np.column_stack([x_world, y_world]))
    return lines


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        prog=PROG,
        description="PlyForge - generate 1:1 ply cutting templates for "
                    "composite layups by non-planar slicing of a mold or "
                    "finished-part STL. Layers start at the top and bottom "
                    "surfaces and work inwards toward the middle.",
    )
    p.add_argument("stl", help="Input STL file (e.g. stabilizer.stl)")
    p.add_argument("--layer-height", type=float, default=0.2,
                   help="Layer thickness in mm (default: 0.2)")
    p.add_argument("--out-dir", default="layers",
                   help="Output directory for PNG files (default: layers)")
    p.add_argument("--resolution", type=int, default=300,
                   help="XY grid resolution for ray casting (default: 300)")
    p.add_argument("--dpi", type=int, default=200,
                   help="Pixel density of the PNGs (default: 200).")

    p.add_argument("--print-scale", dest="print_scale",
                   action="store_true", default=True,
                   help="Render at 1:1 physical scale (default: on)")
    p.add_argument("--no-print-scale", dest="print_scale",
                   action="store_false",
                   help="Disable 1:1 physical scale")

    p.add_argument("--label-step-x", type=float, default=5.0,
                   help="Labeled tick spacing on the X axis in mm "
                        "(default: 5). Use 0 for auto (~20 labels).")
    p.add_argument("--label-step-y", type=float, default=5.0,
                   help="Labeled tick spacing on the Y axis in mm "
                        "(default: 5). Use 0 for auto (~20 labels).")

    p.add_argument("--reference", dest="reference",
                   action="store_true", default=True,
                   help="Draw the reference layer outline (default: on)")
    p.add_argument("--no-reference", dest="reference",
                   action="store_false",
                   help="Disable the reference layer outline")
    p.add_argument("--reference-layer", type=int, default=1,
                   help="Reference layer number (1 = outermost, default: 1)")

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    loaded = trimesh.load(args.stl)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh.vertices.size == 0:
        raise SystemExit("Mesh could not be loaded or is empty.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    bounds = (xmin, xmax, ymin, ymax)

    print(f"Mesh bounds: {mesh.bounds.tolist()}")
    print(f"Building height fields at resolution {args.resolution} ...")

    X, Y, top, bot, valid = build_height_fields(mesh, args.resolution)

    xs = X[0, :]
    ys = Y[:, 0]
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    cos_top = cos_slope(top, dx, dy)
    cos_bot = cos_slope(bot, dx, dy)

    max_thickness = float(np.nanmax(top - bot))
    print(f"Max thickness: {max_thickness:.3f} mm")

    h = args.layer_height
    max_layers = int(np.ceil(max_thickness / (2.0 * h))) + 1
    print(f"Generating up to {max_layers} layers from each side.")

    top_ref_lines = []
    bot_ref_lines = []

    if args.reference:
        ref_idx = max(1, args.reference_layer) - 1

        F_ref = top - (ref_idx + 1) * h * cos_top - bot
        if np.nanmax(np.where(valid, F_ref, -np.inf)) >= 0:
            top_ref_lines = extract_contours(F_ref, 0.0, valid, X, Y)
            print(f"Using top layer {ref_idx + 1} as top reference.")
        else:
            print(f"Top reference layer {ref_idx + 1} is beyond the solid.")

        G_ref = top - (bot + (ref_idx + 1) * h * cos_bot)
        if np.nanmax(np.where(valid, G_ref, -np.inf)) >= 0:
            bot_ref_lines = extract_contours(G_ref, 0.0, valid, X, Y)
            print(f"Using bottom layer {ref_idx + 1} as bottom reference.")
        else:
            print(f"Bottom reference layer {ref_idx + 1} is beyond the solid.")

    for k in range(max_layers):
        Fk = top - (k + 1) * h * cos_top - bot
        if np.nanmax(np.where(valid, Fk, -np.inf)) < 0:
            break

        lines = extract_contours(Fk, 0.0, valid, X, Y)
        if not lines:
            continue

        layer_number = k + 1
        title = f"Top layer {layer_number}"
        out = os.path.join(args.out_dir, f"top_{layer_number:04d}.png")
        write_png(lines, out, bounds, title,
                  reference=top_ref_lines if args.reference else None,
                  dpi=args.dpi,
                  print_scale=args.print_scale,
                  label_step_x=args.label_step_x,
                  label_step_y=args.label_step_y)
        print(f"  wrote {out}  ({title}, {len(lines)} contour(s))")

        if (k + 1) * h * 2 >= max_thickness:
            break

    for j in range(max_layers):
        Gj = top - (bot + (j + 1) * h * cos_bot)
        if np.nanmax(np.where(valid, Gj, -np.inf)) < 0:
            break

        lines = extract_contours(Gj, 0.0, valid, X, Y)
        if not lines:
            continue

        layer_number = j + 1
        title = f"Bottom layer {layer_number}"
        out = os.path.join(args.out_dir, f"bottom_{layer_number:04d}.png")
        write_png(lines, out, bounds, title,
                  reference=bot_ref_lines if args.reference else None,
                  dpi=args.dpi,
                  print_scale=args.print_scale,
                  label_step_x=args.label_step_x,
                  label_step_y=args.label_step_y)
        print(f"  wrote {out}  ({title}, {len(lines)} contour(s))")

        if (j + 1) * h * 2 >= max_thickness:
            break

    print("Done.")


if __name__ == "__main__":
    main()