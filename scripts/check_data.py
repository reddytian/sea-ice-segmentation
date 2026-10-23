#!/usr/bin/env python3
"""Smoke-test the cache + datasets on real data before any training.

Prints: crop throughput, how often a crop has no SOD/FLOE labels (how often
the NaN guard in losses.masked_ce matters), train-split class weights, and
the eval tile counts for val/test.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.data import EvalTiles, TrainCrops, class_weights  # noqa: E402

FILL = 255


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/dl-local/ai4arctic/cache_x5")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = TrainCrops(args.cache, size=args.size, samples=args.n)
    print(f"train: {len(ds.store.scenes)} scenes, {len(ds.store.channels)} channels")
    t0 = time.time()
    v_sic, no_sod, no_floe = [], 0, 0
    for i in range(args.n):
        x, y = ds[i]
        v_sic.append(float((y[0] != FILL).float().mean()))
        no_sod += bool((y[1] == FILL).all())
        no_floe += bool((y[2] == FILL).all())
    dt = time.time() - t0
    print(f"{args.n} crops of {args.size}: {dt:.1f}s ({args.n/dt:.0f} crops/s, single process)")
    print(f"SIC-valid fraction per crop: median {np.median(v_sic):.2f}, "
          f"<0.2 in {np.mean(np.array(v_sic) < 0.2)*100:.0f}% of crops")
    print(f"crops with NO SOD labels: {no_sod/args.n*100:.0f}%   NO FLOE labels: {no_floe/args.n*100:.0f}%")
    print(f"x stats over crops: mean {float(x.mean()):+.2f}, std {float(x.std()):.2f} (last crop)")

    cw = class_weights(args.cache)
    for t, w in cw.items():
        print(f"class weights {t:4s}: " + " ".join(f"{v:.2f}" for v in w.tolist()))

    for split in ("val", "test"):
        ev = EvalTiles(args.cache, split, tile=512)
        print(f"{split}: {len(ev.store.scenes)} scenes -> {len(ev)} tiles of 512")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
