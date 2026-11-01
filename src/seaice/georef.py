"""Radar-geometry class maps -> map-projected Cloud-Optimised GeoTIFFs.

AI4Arctic scenes are in Sentinel-1 swath geometry (lines x samples), located by
lat/lon tie points. The model's 400 m grid is that swath block-averaged by
`factor`, so a full-resolution tie point at (line L, sample S) sits at
((L + 0.5) / f, (S + 0.5) / f) in the model grid (GDAL pixel-corner convention).

Warping uses GDAL's thin-plate-spline GCP transformer. On a synthetic curved
swath with tie points every 8 output pixels (tests/test_predict.py) it places
pixels with a median error of 0.40 and a 99th percentile of 0.68 source pixels,
with no systematic shift (|mean| < 0.01 px) -- the limit of nearest-neighbour
resampling. A global polynomial fit (GDAL's default for GCPs) fails that test, and
on a 400 km prototype swath it was off by ~25 pixels.
Class maps are resampled nearest-neighbour; categorical values are never blended.
"""
from __future__ import annotations

import logging

import numpy as np


class _QuietSrcMethod(logging.Filter):
    """rasterio hands SRC_METHOD to both the GCP transformer (where it takes effect)
    and GDAL's warp options (which ignore it and warn). Drop that one warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "SRC_METHOD" not in record.getMessage()


logging.getLogger("rasterio._env").addFilter(_QuietSrcMethod())

MAX_GCPS = 600          # thin-plate-spline cost grows ~cubically with the GCP count
# How the sar_grid2d_latitude/longitude arrays map onto pixels -- measured on all 123
# scenes by scripts/check_geolocation.py against an independent coastline and against
# geodesic node spacing:
#   * axis 0 runs along LINES, axis 1 along SAMPLES, neither reversed, whatever the
#     scene's `flip` attribute (the dimension names say the opposite order);
#   * line nodes sit every 250 lines from line 0, plus one node on the last line
#     (Sentinel-1 annotation spacing: 500 lines at 40 m), so the node count varies
#     with scene length -- 21 only for ~5000-line scenes;
#   * the 21 sample nodes are evenly spaced from the first to the last sample.
# Treating line nodes as evenly spaced would misplace rows by > 1 km in 38 of 123 scenes (up to 16 km).
GRID_CONVENTION = (True, False, False)          # (swap, rev_lines, rev_samples)
LINE_NODE_STEP = 250


def grid_to_line_sample(g: np.ndarray, swap: bool, rev_lines: bool, rev_samples: bool) -> np.ndarray:
    """A stored sar_grid2d_* array -> the same grid indexed [line node, sample node].

    `swap=False` reads the stored dims as named, (sample, line); `swap=True` as
    (line, sample). `rev_*` reverse an axis relative to the pixel arrays.
    """
    g = np.asarray(g) if swap else np.asarray(g).T
    if rev_lines:
        g = g[::-1]
    if rev_samples:
        g = g[:, ::-1]
    return g


def node_positions(n_line_nodes: int, n_sample_nodes: int, n_lines: int,
                   n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution pixel index of every grid row (lines) and column (samples)."""
    lines = np.r_[LINE_NODE_STEP * np.arange(n_line_nodes - 1, dtype=float), n_lines - 1.0]
    if not (lines[-2] < lines[-1] <= lines[-2] + 1.5 * LINE_NODE_STEP):
        raise ValueError(f"{n_line_nodes} line nodes do not fit {n_lines} lines at a "
                         f"{LINE_NODE_STEP}-line spacing")
    samples = np.arange(n_sample_nodes) * (n_samples - 1) / (n_sample_nodes - 1)
    return lines, samples


def to_gcps(lines, samples, lats, lons, factor: int, crs: str = "EPSG:3413"):
    """Tie points (full-resolution line/sample, degrees) -> GCPs in `crs` on the model grid.

    GCPs are projected to the target CRS first: the spline is fitted where the
    output lives, and lat/lon axis-order and pole issues cannot arise.
    """
    from rasterio.control import GroundControlPoint
    from rasterio.warp import transform

    lines, samples = np.asarray(lines, float).ravel(), np.asarray(samples, float).ravel()
    lats, lons = np.asarray(lats, float).ravel(), np.asarray(lons, float).ravel()
    ok = np.isfinite(lines) & np.isfinite(samples) & np.isfinite(lats) & np.isfinite(lons)
    lines, samples, lats, lons = lines[ok], samples[ok], lats[ok], lons[ok]
    if len(lines) < 10:
        raise ValueError(f"only {len(lines)} valid tie points")
    if len(lines) > MAX_GCPS:
        # Thin spatially: one tie point per cell of a coarse line/sample grid, plus the
        # extremes, so the spline still spans the whole swath.
        nb = int(np.sqrt(MAX_GCPS)) - 1
        bl = np.minimum(((lines - lines.min()) / np.ptp(lines) * nb).astype(int), nb - 1)
        bs = np.minimum(((samples - samples.min()) / np.ptp(samples) * nb).astype(int), nb - 1)
        _, first = np.unique(bl * nb + bs, return_index=True)
        keep = np.unique(np.r_[first, np.argmin(lines), np.argmax(lines),
                               np.argmin(samples), np.argmax(samples)])
        lines, samples, lats, lons = lines[keep], samples[keep], lats[keep], lons[keep]
    xs, ys = transform("EPSG:4326", crs, lons.tolist(), lats.tolist())
    # z must be a number: rasterio writes z=None as the text "None", and GDAL then
    # rejects every GCP ("Unable to compute a transformation").
    return [GroundControlPoint(row=(l + 0.5) / factor, col=(s + 0.5) / factor, x=x, y=y, z=0.0)
            for l, s, x, y in zip(lines, samples, xs, ys)]


def warp(arrays: dict[str, np.ndarray], gcps, crs: str = "EPSG:3413", res: float = 400.0,
         nodata: int = 255):
    """Warp same-shaped uint8 arrays onto one north-up grid. Returns (warped dict, transform)."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    from rasterio.transform import from_origin

    h, w = next(iter(arrays.values())).shape
    opts = {"SRC_METHOD": "GCP_TPS"}
    tr, dw, dh = calculate_default_transform(crs, crs, w, h, gcps=gcps, resolution=res, **opts)
    # Snap to a global grid of `res` (like gdalwarp -tap): every scene's pixels then
    # line up, so maps of the same area from different dates compare pixel for pixel.
    left, top = np.floor(tr.c / res) * res, np.ceil(tr.f / res) * res
    right, bottom = np.ceil((tr.c + dw * res) / res) * res, np.floor((tr.f - dh * res) / res) * res
    dw, dh = int(round((right - left) / res)), int(round((top - bottom) / res))
    tr = from_origin(left, top, res, res)
    out = {}
    for name, a in arrays.items():
        dst = np.full((dh, dw), nodata, dtype=np.uint8)
        reproject(a, dst, gcps=gcps, src_crs=crs, src_nodata=nodata, dst_transform=tr,
                  dst_crs=crs, dst_nodata=nodata, resampling=Resampling.nearest, **opts)
        out[name] = dst
    return out, tr


def write_cog(path, a: np.ndarray, transform, crs: str, nodata: int = 255,
              colormap: dict[int, tuple] | None = None, description: str = "",
              tags: dict[str, str] | None = None) -> None:
    """Single-band uint8 COG: DEFLATE, 512 px tiles, MODE overviews (categorical), colour table.

    Built in memory and written out in one go: GDAL's COG driver otherwise creates and
    deletes temporary overview files next to the target, which fails on file systems
    that forbid deletion and leaves half-written files if interrupted.
    """
    from rasterio.io import MemoryFile

    prof = dict(driver="COG", width=a.shape[1], height=a.shape[0], count=1, dtype="uint8",
                crs=crs, transform=transform, nodata=nodata, compress="DEFLATE",
                blocksize=512, overview_resampling="MODE")
    with MemoryFile() as mem:
        with mem.open(**prof) as dst:
            dst.write(a, 1)
            if colormap:
                dst.write_colormap(1, {**colormap, nodata: (0, 0, 0, 0)})
            if description:
                dst.set_band_description(1, description)
            if tags:
                dst.update_tags(1, **tags)
        data = mem.read()
    with open(path, "wb") as fh:
        fh.write(data)
