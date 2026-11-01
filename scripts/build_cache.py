#!/usr/bin/env python3
"""Downsample, normalise and cache every AI4Arctic scene as memory-mappable .npy.

Output layout (default ~/dl-local/ai4arctic/cache_x5/):
    x/<scene>.npy   float16 (C, H, W)  normalised inputs, NaN -> 0
    y/<scene>.npy   uint8   (3, H, W)  SIC, SOD, FLOE; 255 = fill
    index.json      channels, normalisation stats, per-scene split/shape/valid fractions

Choices and why:
  * SAR is block-averaged by `factor` (default 5: 80 m -> 400 m).
  * Labels take the CENTRE pixel of each block, not the block majority. A
    majority vote erodes small classes along every boundary; centre sampling
    keeps class proportions unbiased in expectation.
  * AMSR2 is resampled with a FIXED 25-SAR-pixel spacing. Across the corpus,
    the 2 km grid has floor(N_sar / 25) cells in 118/123 scenes (one fewer in
    the rest), leaving 0-25 SAR pixels at the far edge with no AMSR cell.
    Stretching AMSR over the full SAR extent would shift it by up to one
    whole 2 km cell at that edge. Cell k is assumed centred at SAR pixel
    25k + 12 (--amsr-center); pixels past the last cell take its value.
  * Normalisation mean/std come from TRAIN regions only, over pixels with a
    valid SIC label. Val/test never influence the inputs.
  * Two passes: pass 1 = train statistics, pass 2 = write. Files are written
    to a temp name and renamed, so an interrupted run leaves no half files.

The per-scene transforms live in src/seaice/preprocess.py and are shared with
inference (seaice.predict), so a deployed model sees exactly these inputs.

Run with --dry-run first: it prints the channels picked from the first scene
and stops, so a wrong variable guess costs seconds, not a full pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    sys.exit("needs xarray: pip install xarray netcdf4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.preprocess import (AMSR_DIMS, AMSR_SPACING, FILL, LABELS, SAR_DIMS,  # noqa: E402,F401
                               amsr_to_grid, block_centre, block_mean, open_scene,
                               normalise, scene_inputs, scene_labels)
from seaice.preprocess import pick_channels as _pick_channels  # noqa: E402


# ---------------------------------------------------------------- helpers
def load_regions(manifest: Path) -> dict[str, str]:
    reg = {}
    with open(manifest) as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 4 and p[0] != "file":
                reg[p[0]] = p[3]
    return reg


def pick_channels(ds) -> tuple[list[str], list[str]]:
    try:
        return _pick_channels(ds)
    except ValueError as e:
        sys.exit(str(e))


def build_scene(path: Path, sar_vars, amsr_vars, f: int, centre: float):
    """Return x (C, H, W) float32 un-normalised, y (3, H, W) uint8."""
    ds = open_scene(path)
    try:
        x = scene_inputs(ds, list(sar_vars) + list(amsr_vars), f, centre)
        y = scene_labels(ds, f)
    finally:
        ds.close()
    assert y.shape[1:] == x.shape[1:], (y.shape, x.shape)
    return x, y


def atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="~/dl-local/ai4arctic/scenes")
    ap.add_argument("--manifest", default="~/dl-local/ai4arctic/manifest.csv")
    ap.add_argument("--split", default="configs/split_region_v1.json")
    ap.add_argument("--out", default="~/dl-local/ai4arctic/cache_x5")
    ap.add_argument("--factor", type=int, default=5)
    ap.add_argument("--amsr-center", type=float, default=12.0,
                    help="SAR pixel index of AMSR cell 0's centre")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ex = lambda p: Path(os.path.expanduser(p))

    scene_dir, out = ex(args.scenes), ex(args.out)
    files = sorted(scene_dir.glob("*.nc"))
    if not files:
        sys.exit(f"no .nc files in {scene_dir}")

    regions = load_regions(ex(args.manifest))
    split_cfg = json.loads(ex(args.split).read_text())
    region_split = {r: s for s in ("train", "val", "test") for r in split_cfg[s]}
    scene_split = {}
    for p in files:
        r = regions.get(p.name)
        if r is None or r not in region_split:
            sys.exit(f"{p.name}: region {r!r} not in manifest/split")
        scene_split[p.name] = region_split[r]
    counts = {s: sum(v == s for v in scene_split.values()) for s in ("train", "val", "test")}
    if counts != split_cfg["expected_scenes"]:
        sys.exit(f"scene counts {counts} != expected {split_cfg['expected_scenes']}")

    with xr.open_dataset(files[0], mask_and_scale=False) as ds0:
        sar_vars, amsr_vars = pick_channels(ds0)
        print(f"first scene {files[0].name}")
        for v in sar_vars + amsr_vars:
            print(f"  {v:28s} {str(ds0[v].dims):40s} {ds0[v].dtype}")
        sz = dict(ds0.sizes)
    channels = sar_vars + amsr_vars
    H, W = sz["sar_lines"] // args.factor, sz["sar_samples"] // args.factor
    est = len(files) * (len(channels) * 2 + 3) * H * W / 1e9
    print(f"{len(channels)} input channels, factor {args.factor} -> ~{H}x{W} per scene,"
          f" ~{est:.1f} GB on disk")
    print(f"split: {counts}")
    if args.dry_run:
        x, y = build_scene(files[0], sar_vars, amsr_vars, args.factor, args.amsr_center)
        print(f"dry run built x{x.shape} y{y.shape}; "
              f"x range per channel min {np.nanmin(x, axis=(1, 2)).round(2).tolist()}")
        print(f"                    max {np.nanmax(x, axis=(1, 2)).round(2).tolist()}")
        print(f"NaN per channel {np.isnan(x).sum(axis=(1, 2)).tolist()}")
        return 0

    # ---- pass 1: train-only statistics over SIC-valid pixels ----
    train_files = [p for p in files if scene_split[p.name] == "train"]
    s1 = np.zeros(len(channels)); s2 = np.zeros(len(channels)); n = np.zeros(len(channels))
    t0 = time.time()
    for i, p in enumerate(train_files, 1):
        x, y = build_scene(p, sar_vars, amsr_vars, args.factor, args.amsr_center)
        m = y[0] != FILL
        v = x[:, m].astype(np.float64)
        ok = ~np.isnan(v)
        s1 += np.where(ok, v, 0).sum(1); s2 += np.where(ok, v * v, 0).sum(1); n += ok.sum(1)
        print(f"\r  pass 1 (train stats) {i:3d}/{len(train_files)}  {time.time()-t0:6.0f}s",
              end="", flush=True)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    if (std < 1e-6).any():
        sys.exit(f"constant channel(s): {[c for c, s in zip(channels, std) if s < 1e-6]}")
    print()

    # ---- pass 2: write every scene ----
    (out / "x").mkdir(parents=True, exist_ok=True)
    (out / "y").mkdir(parents=True, exist_ok=True)
    scenes = []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        x, y = build_scene(p, sar_vars, amsr_vars, args.factor, args.amsr_center)
        n_nan = int(np.isnan(x).sum())
        xn = normalise(x, mean, std)
        stem = p.name[:-3]
        atomic_save(out / "x" / f"{stem}.npy", xn)
        atomic_save(out / "y" / f"{stem}.npy", y)
        scenes.append({
            "name": stem, "region": regions[p.name], "split": scene_split[p.name],
            "shape": list(y.shape[1:]), "nan_inputs": n_nan,
            "valid_frac": {l: round(float((y[k] != FILL).mean()), 5)
                           for k, l in enumerate(LABELS)},
        })
        print(f"\r  pass 2 (write) {i:3d}/{len(files)}  {time.time()-t0:6.0f}s",
              end="", flush=True)
    print()

    index = {
        "version": "cache_v1",
        "factor": args.factor,
        "amsr_spacing": AMSR_SPACING,
        "amsr_center": args.amsr_center,
        "label_sampling": "block centre",
        "channels": channels,
        "norm": {"mean": mean.tolist(), "std": std.tolist(),
                 "from": "train regions, SIC-valid pixels",
                 "split_file": str(args.split)},
        "fill": FILL,
        "labels": list(LABELS),
        "scenes": scenes,
    }
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    tot = sum(f.stat().st_size for f in out.rglob("*.npy")) / 1e9
    print(f"wrote {len(scenes)} scenes, {tot:.2f} GB -> {out}")
    nan_sc = [s["name"] for s in scenes if s["nan_inputs"]]
    if nan_sc:
        print(f"NOTE: {len(nan_sc)} scenes had NaN inputs (set to 0 = channel mean): {nan_sc[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
