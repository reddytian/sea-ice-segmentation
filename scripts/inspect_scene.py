#!/usr/bin/env python3
"""Print what one AI4Arctic scene holds beyond the variables the model reads.

Needed before writing georeferenced output: how the scene's geolocation is stored
(tie-point grids? full-resolution lat/lon?), how invalid / land pixels are marked,
and which global attributes describe the scene.

    python scripts/inspect_scene.py                 # a test-region scene
    python scripts/inspect_scene.py 20190130T     # any scene whose file name matches
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

SAR_DIMS = ("sar_lines", "sar_samples")
AMSR_DIMS = ("2km_grid_lines", "2km_grid_samples")
GEO_WORDS = ("lat", "lon", "grid", "line", "sample", "height", "time", "crs", "proj")


def short(v, n=90):
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def stats(a):
    a = np.asarray(a)
    if a.dtype.kind not in "fiu":
        return f"dtype {a.dtype}"
    f = a.astype(np.float64)
    ok = np.isfinite(f)
    if not ok.any():
        return "all non-finite"
    return f"min {np.min(f[ok]):.6g}  max {np.max(f[ok]):.6g}  non-finite {int((~ok).sum())}"


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "20210421T082208"
    root = Path(os.path.expanduser("~/dl-local/ai4arctic/scenes"))
    files = sorted(p for p in root.glob("*.nc") if pattern in p.name)
    if not files:
        sys.exit(f"no scene matching {pattern!r} in {root}")
    p = files[0]
    print(f"file {p.name}  {p.stat().st_size / 1e6:.0f} MB")

    ds = xr.open_dataset(p, mask_and_scale=False)
    print(f"dims {dict(ds.sizes)}")
    print(f"coords {list(ds.coords)}")
    print("global attrs:")
    for k, v in ds.attrs.items():
        print(f"  {k}: {short(v)}")

    print("variables NOT on the SAR or 2 km grid, or with a geolocation-like name:")
    for name, v in ds.variables.items():
        geo_name = any(w in name.lower() for w in GEO_WORDS)
        if v.dims in (SAR_DIMS, AMSR_DIMS) and not geo_name:
            continue
        a = v.values
        print(f"  {name:30s} {str(v.dims):45s} {str(a.shape):16s} {a.dtype}")
        attrs = {k: short(v.attrs[k], 60) for k in ("long_name", "units", "_FillValue",
                                                      "standard_name") if k in v.attrs}
        if attrs:
            print(f"      attrs {attrs}")
        print(f"      {stats(a)}")
        if a.ndim == 1:
            print(f"      first {a[:6].tolist()}  last {a[-3:].tolist()}")
        elif a.ndim == 2 and a.dtype.kind in "fiu":
            print(f"      corners [0,0]={a[0, 0]} [0,-1]={a[0, -1]} [-1,0]={a[-1, 0]} "
                  f"[-1,-1]={a[-1, -1]}")

    hh = ds["nersc_sar_primary"]
    a = hh.values
    print(f"nersc_sar_primary attrs {dict((k, short(x, 40)) for k, x in hh.attrs.items())}")
    print(f"  {stats(a)}  corners {a[0, 0]} {a[0, -1]} {a[-1, 0]} {a[-1, -1]}")
    sic = ds["SIC"].values
    sar_bad = ~np.isfinite(a)
    print(f"SIC fill (255) {np.mean(sic == 255):.3f} of pixels; "
          f"SAR non-finite {sar_bad.mean():.3f}; "
          f"SIC fill where SAR non-finite {np.mean(sic[sar_bad] == 255) if sar_bad.any() else 'n/a'}")

    for name in ds.data_vars:
        if "dist" in name.lower() or "land" in name.lower() or "mask" in name.lower():
            d = ds[name].values
            vals, counts = np.unique(d, return_counts=True)
            print(f"{name} {ds[name].dims} attrs {dict((k, short(x, 50)) for k, x in ds[name].attrs.items())}")
            print(f"  {len(vals)} distinct values; first {vals[:8].tolist()} last {vals[-3:].tolist()}")
            if d.shape == sic.shape:
                for val in vals[:3]:
                    m = d == val
                    print(f"  value {val}: {m.mean():.3f} of pixels, SIC fill there "
                          f"{np.mean(sic[m] == 255):.3f}, SAR non-finite there {np.mean(sar_bad[m]):.3f}")
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
