"""A small synthetic scene in the AI4Arctic ready-to-train layout, with a known
swath geometry so that georeferencing can be checked against ground truth.

Used by tests/test_predict.py and by the Docker smoke test in CI, which passes the
real model's channel list so the published bundle can run on it:
    python tests/fake_scene.py OUT.nc [--channels nersc_sar_primary,...,btemp_89_0v]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SAR = ("sar_lines", "sar_samples")
AMSR = ("2km_grid_lines", "2km_grid_samples")
CHANNELS = ["nersc_sar_primary", "nersc_sar_secondary", "sar_incidenceangle",
            "btemp_18_7h", "btemp_36_5v", "btemp_6_9h", "btemp_89_0v"]

# Swath geometry in EPSG:3413 metres, at full (80 m) resolution: rotated by 25 deg,
# with gentle curvature plus a smooth non-polynomial term (~2 output pixels), so a
# global polynomial fit cannot be exact while tie points still sample it densely.
X0, Y0, THETA, PIX = -200_000.0, -2_000_000.0, np.deg2rad(25), 80.0


def swath_xy(line, sample):
    """Full-resolution (line, sample) pixel coordinates -> EPSG:3413 (x, y)."""
    u, v = PIX * np.asarray(sample, float), PIX * np.asarray(line, float)
    x = X0 + u * np.cos(THETA) - v * np.sin(THETA) + 1e-6 * (u - 2e4) ** 2 \
        + 800 * np.sin(2 * np.pi * v / 16e3)
    y = Y0 - u * np.sin(THETA) - v * np.cos(THETA) + 8e-7 * (v - 2e4) ** 2 \
        + 800 * np.sin(2 * np.pi * u / 16e3)
    return x, y


LAND = (slice(950, None), slice(420, None))    # a land patch in one corner (lines, samples)


def write(path, H: int = 1130, W: int = 520, seed: int = 0, channels: list[str] | None = None) -> None:
    """Scene in the real file layout: SAR stored as 0 outside the swath, land marked in
    distance_map, and the lat/lon grid as AI4Arctic stores it -- one row every 250 lines
    plus the last line (6 rows here, the last step short), 21 even columns, axis 0 = lines
    under dimension names that say samples."""
    import xarray as xr
    from rasterio.warp import transform

    from seaice.preprocess import LAND_VALUE

    rng = np.random.default_rng(seed)
    channels = channels or CHANNELS
    v = {}
    h2, w2 = H // 25, W // 25
    for name in channels:
        if name.startswith("nersc_sar"):                # backscatter; 0 = no data (swath edge)
            a = rng.normal(0.5 if name.endswith("primary") else -0.5, 1, (H, W)).astype("f4")
            a[:, :23] = 0.0
            v[name] = (SAR, a)
        elif "incidence" in name:
            v[name] = (SAR, np.broadcast_to(np.linspace(-1.7, 1.7, W, dtype="f4"), (H, W)).copy())
        else:                                           # everything else: 2 km grid
            v[name] = (AMSR, rng.normal(0, 1, (h2, w2)).astype("f4"))
    dm = np.full((H, W), -1.0, "f4")
    dm[LAND] = LAND_VALUE
    v["distance_map"] = (SAR, dm)
    for name, n in (("SIC", 11), ("SOD", 6), ("FLOE", 7)):
        a = rng.integers(0, n, (H, W)).astype("u1")
        a[:, :23] = 255
        a[LAND] = 255
        v[name] = (SAR, a)
    # Written as AI4Arctic files are, independently of seaice.georef, so that a wrong
    # reading convention in the code fails the tests instead of cancelling out.
    kl = np.r_[np.arange(0, H - 1, 250), H - 1]
    ks = np.arange(21) * (W - 1) / 20
    ll, ss = np.meshgrid(kl, ks, indexing="ij")
    x, y = swath_xy(ll + 0.5, ss + 0.5)                 # node k = centre of pixel k
    lon, lat = transform("EPSG:3413", "EPSG:4326", x.ravel().tolist(), y.ravel().tolist())
    dims = ("sar_sample_2dgrid_points", "sar_line_2dgrid_points")   # sic: axis 0 holds lines
    v["sar_grid2d_latitude"] = (dims, np.reshape(lat, ll.shape))
    v["sar_grid2d_longitude"] = (dims, np.reshape(lon, ll.shape))
    xr.Dataset(v, attrs={"flip": np.nan, "pixel_spacing": 80}).to_netcdf(path)  # flip is NaN in some real files


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--channels", help="comma-separated input variables (default: a 7-channel set)")
    a = ap.parse_args()
    write(a.out, channels=a.channels.split(",") if a.channels else None)
    print(a.out)
