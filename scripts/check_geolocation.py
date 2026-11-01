#!/usr/bin/env python3
"""Check how AI4Arctic's sar_grid2d_latitude/longitude map onto pixels, against an
independent coastline. This is the evidence behind seaice.georef.GRID_CONVENTION.

The files do not document it: the grid has a `flip` attribute (sometimes NaN),
dimension names, and no line/sample index per node. A wrong reading mirrors,
rotates or stretches every output map. The test: each scene's own land mask
(distance_map id 0) must coincide with an independent 1 km land mask (GLOBE, via
the global-land-mask package) once pixels are located with the right reading.

Per scene it scores, by land IoU:
  * all 8 axis readings (2 orders x reversed lines x reversed samples);
  * the adopted line-node layout (every 250 lines + last line) against evenly
    spaced line nodes;
  * shifts of up to +-24 pixels (+-1.9 km) of the adopted reading, to bound
    what misregistration remains. GLOBE is ~1 km, so ~10 px is its noise floor.

    pip install global-land-mask
    python scripts/check_geolocation.py                    # all scenes -> docs/geolocation_check.jsonl
    python scripts/check_geolocation.py --first 60         # resume / run in chunks (appends)
    python scripts/check_geolocation.py --summary docs/geolocation_check.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.georef import GRID_CONVENTION, grid_to_line_sample, node_positions  # noqa: E402
from seaice.preprocess import LAND_VALUE, open_scene  # noqa: E402

STEP = 12                                 # evaluate every 12th pixel (~1 km, the reference's scale)
SHIFTS = (-24, -16, -8, 0, 8, 16, 24)


def name(c) -> str:
    swap, rev_l, rev_s = c
    return ("axis0=line" if swap else "axis0=sample") + (" rev-lines" if rev_l else "") + \
           (" rev-samples" if rev_s else "")


def bilinear(g, fl, fs):
    n0, n1 = g.shape
    l0, s0 = np.minimum(fl.astype(int), n0 - 2), np.minimum(fs.astype(int), n1 - 2)
    a, b = fl - l0, fs - s0
    return ((g[l0, s0] * (1 - b) + g[l0, s0 + 1] * b) * (1 - a)
            + (g[l0 + 1, s0] * (1 - b) + g[l0 + 1, s0 + 1] * b) * a)


def land_iou(lat_g, lon_g, lpos, spos, L, S, land, shift=(0, 0)) -> float:
    """Land IoU if grid row k sits at line lpos[k] + shift[0], column j at spos[j] + shift[1]."""
    from global_land_mask import globe
    fl = np.interp(L - shift[0], lpos, np.arange(len(lpos)), left=np.nan, right=np.nan)
    fs = np.interp(S - shift[1], spos, np.arange(len(spos)), left=np.nan, right=np.nan)
    ok = np.isfinite(fl) & np.isfinite(fs)
    lat, lon = bilinear(lat_g, fl[ok], fs[ok]), bilinear(lon_g, fl[ok], fs[ok])
    ref = globe.is_land(lat, (lon + 180) % 360 - 180)
    m = land[ok]
    return float(np.sum(ref & m) / max(np.sum(ref | m), 1))


def check(path: Path) -> dict:
    ds = open_scene(path)
    try:
        lat_s = np.asarray(ds["sar_grid2d_latitude"].values, float)
        lon_s = np.asarray(ds["sar_grid2d_longitude"].values, float)
        nl, ns = ds.sizes["sar_lines"], ds.sizes["sar_samples"]
        dm = np.asarray(ds["distance_map"].values[::STEP, ::STEP], float)
        row = {"scene": path.name[:15], "service": ds.attrs.get("ice_service"),
               "flip": str(ds.attrs.get("flip")), "lines": nl, "grid": list(lat_s.shape)}
    finally:
        ds.close()
    land = np.abs(dm - LAND_VALUE) < 1e-4
    row["land"] = round(float(land.mean()), 4)
    if land.mean() < 0.005:
        return row
    L, S = np.meshgrid(np.arange(dm.shape[0]) * STEP, np.arange(dm.shape[1]) * STEP, indexing="ij")

    def score(c, even_lines=False, shift=(0, 0)):
        la, lo = grid_to_line_sample(lat_s, *c), grid_to_line_sample(lon_s, *c)
        try:
            lpos, spos = node_positions(*la.shape, nl, ns)
        except ValueError:                          # this reading does not even fit the shape
            return 0.0
        if even_lines:
            lpos = np.arange(la.shape[0]) * (nl - 1) / (la.shape[0] - 1)
        return land_iou(la, lo, lpos, spos, L, S, land, shift)

    readings = {name(c): round(score(c), 4) for c in itertools.product((False, True), repeat=3)}
    row["readings"] = readings
    row["best_reading"] = max(readings, key=readings.get)
    row["iou"] = readings[name(GRID_CONVENTION)]
    row["iou_even_lines"] = round(score(GRID_CONVENTION, even_lines=True), 4)
    shifts = {(a, b): score(GRID_CONVENTION, shift=(a, b)) for a in SHIFTS for b in SHIFTS}
    best = max(shifts, key=shifts.get)
    row["best_shift_px"], row["iou_best_shift"] = list(best), round(shifts[best], 4)
    return row


def summary(rows: list[dict]) -> None:
    rows = [r for r in rows if "iou" in r]
    print(f"\n{len(rows)} scenes with >= 0.5% land")
    wins = defaultdict(int)
    for r in rows:
        wins[r["best_reading"]] += 1
    print("best axis reading:", dict(wins))
    iou = np.array([r["iou"] for r in rows])
    even = np.array([r["iou_even_lines"] for r in rows])
    print(f"land IoU, adopted reading: median {np.median(iou):.3f}, 10th pct {np.percentile(iou, 10):.3f}")
    gain = iou - even
    print(f"line nodes every 250 lines vs evenly spaced: IoU better by > 0.02 in {np.sum(gain > 0.02)} "
          f"scenes (max +{gain.max():.3f}), worse by > 0.02 in {np.sum(gain < -0.02)}")
    sh = np.array([r["best_shift_px"] for r in rows], float)
    d = np.hypot(sh[:, 0], sh[:, 1]) * 80
    print(f"best-fit residual shift: median {np.median(d):.0f} m, 90th pct {np.percentile(d, 90):.0f} m, "
          f"max {d.max():.0f} m (search limit {max(SHIFTS) * 80 * 1.414:.0f} m)")
    groups = defaultdict(list)
    for r, s in zip(rows, sh):
        groups[(r["service"], r["flip"])].append(s)
    for (svc, fl), v in sorted(groups.items()):
        v = np.array(v)
        se = v.std(0) / np.sqrt(len(v)) if len(v) > 1 else np.full(2, np.nan)
        print(f"  {svc} flip={fl:4s} n={len(v):3d}  mean shift (lines, samples) "
              f"({v[:, 0].mean():+5.1f}, {v[:, 1].mean():+5.1f}) +- ({se[0]:.1f}, {se[1]:.1f}) px")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="~/dl-local/ai4arctic/scenes")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0 = to the end")
    ap.add_argument("--out", default="docs/geolocation_check.jsonl",
                    help="one JSON row per scene; overwritten when --first is 0, else appended")
    ap.add_argument("--summary", help="only summarise an existing JSONL file")
    args = ap.parse_args()
    if args.summary:
        summary([json.loads(x) for x in Path(args.summary).read_text().splitlines() if x.strip()])
        return 0
    try:
        import global_land_mask  # noqa: F401
    except ImportError:
        sys.exit("needs: pip install global-land-mask")
    files = sorted(Path(os.path.expanduser(args.raw)).glob("*.nc"))
    files = files[args.first: args.first + args.count if args.count else None]
    rows = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w" if args.first == 0 else "a") as fh:
        for p in files:
            r = check(p)
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "iou" in r:
                print(f"{r['scene']} {r['service']} flip={r['flip']:4s} land {100 * r['land']:4.1f}%  "
                      f"best {r['best_reading']:12s} IoU {r['iou']:.3f} (even lines {r['iou_even_lines']:.3f})"
                      f"  best shift {tuple(r['best_shift_px'])} -> {r['iou_best_shift']:.3f}")
            else:
                print(f"{r['scene']} {r['service']} flip={r['flip']:4s} land {100 * r['land']:4.1f}%  (too little land)")
    summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
