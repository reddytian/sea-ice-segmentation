#!/usr/bin/env python3
"""Figure of seaice-predict output as delivered: map-projected COGs (EPSG:3413),
with an independent coastline (GLOBE 1 km land mask) drawn on top.

The model never sees the coastline; the masked land / no-data edge of each map
comes from the scene's own land mask and geolocation grid. Where the outline and
that edge agree, the geolocation reading in seaice.georef is right.

    python scripts/map_figure.py predictions/ --out docs/figures/deployed_maps.png
    (needs: pip install matplotlib global-land-mask)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rasterio.warp import transform as warp_xy  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
LAND, COAST = "#d9d7cf", "#3a3936"
SIC_COLORS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
SOD_COLORS = ["#cde2fb", "#ff8e64", "#ea6b3a", "#cd501a", "#aa3b01", "#842c00"]
SOD_NAMES = ["open water", "new", "young", "thin first-year", "thick first-year", "old"]


def land_on_grid(src) -> np.ndarray:
    """GLOBE land mask sampled at every pixel centre of a map-projected raster."""
    from global_land_mask import globe
    rows, cols = np.mgrid[0:src.height, 0:src.width]
    xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
    lon, lat = warp_xy(src.crs, "EPSG:4326", xs, ys)
    return globe.is_land(np.asarray(lat), np.asarray(lon)).reshape(src.height, src.width)


def draw(ax, path: Path, colors, values, land):
    with rasterio.open(path) as src:
        a = src.read(1)
        b = src.bounds
    ext = (b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3)
    ax.imshow(np.where(land, 1, np.nan), cmap=ListedColormap([LAND]), extent=ext,
              interpolation="nearest")
    idx = np.full(a.shape, np.nan)
    for k, v in enumerate(values):
        idx[a == v] = k
    cm = ListedColormap(colors)
    ax.imshow(idx, cmap=cm, norm=BoundaryNorm(np.arange(-0.5, len(colors)), cm.N),
              extent=ext, interpolation="nearest")
    ax.contour(np.flipud(land.astype(float)), levels=[0.5], colors=COAST, linewidths=0.6,
               extent=ext)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e1e0d9")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--out", default="docs/figures/deployed_maps.png")
    args = ap.parse_args()
    metas = sorted(Path(args.pred_dir).glob("*.json"))
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "font.size": 10})
    fig, axes = plt.subplots(len(metas), 2, figsize=(9.5, 4.3 * len(metas) + 0.8), squeeze=False)
    for row, mpath in enumerate(metas):
        m = json.loads(mpath.read_text())
        d = Path(args.pred_dir)
        with rasterio.open(d / m["outputs"]["SIC"]) as src:
            land = land_on_grid(src)
        draw(axes[row, 0], d / m["outputs"]["SIC"], SIC_COLORS, list(range(0, 101, 10)), land)
        draw(axes[row, 1], d / m["outputs"]["SOD"], SOD_COLORS, list(range(6)), land)
        s = m["scene"][:8]
        axes[row, 0].set_ylabel(f"{s[:4]}-{s[4:6]}-{s[6:]}\n{m['scene'][16:19].upper()} chart region",
                                fontsize=10, color=INK2)
        if row == 0:
            axes[row, 0].set_title("Sea ice concentration", fontsize=10, color=INK2)
            axes[row, 1].set_title("Stage of development", fontsize=10, color=INK2)
    fig.suptitle("seaice-predict output on held-out test scenes: EPSG:3413, 400 m; "
                 "outline = independent GLOBE coastline", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=1.0)
    sic = [Patch(color=SIC_COLORS[k], label=f"{k * 10}%") for k in (0, 2, 4, 6, 8, 10)]
    sod = [Patch(color=c, label=n) for c, n in zip(SOD_COLORS, SOD_NAMES)]
    extra = [Patch(color=LAND, label="land (GLOBE)")]
    fig.legend(handles=sic + extra, loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=4,
               frameon=False, fontsize=9, title="SIC", title_fontsize=9)
    fig.legend(handles=sod, loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=3,
               frameon=False, fontsize=9, title="SOD", title_fontsize=9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.15)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
