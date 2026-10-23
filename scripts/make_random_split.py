#!/usr/bin/env python3
"""The control arm of the leakage experiment: a RANDOM SCENE split.

dl-learning Day 11 measured, on synthetic data, how much a random split inflates
validation scores when samples are spatially correlated. This is the same test
on real data, where the correlation is physical: scenes from one region share an
ice regime, an ice service's charting conventions, and often ice days apart.

Both arms share the SAME test scenes (the three held-out test regions):
    region arm  train 10 regions / val 3 other regions   (configs/split_region_v1.json)
    random arm  the 99 non-test scenes shuffled into train/val, same sizes
Each arm selects its checkpoint on its own val, then scores test once.
optimism = val score - test score. The random arm's val shares regions with its
train set; the region arm's does not. Test is unseen ground for both.

One small asymmetry, stated rather than hidden: the cache was normalised with
mean/std from the region arm's 75 train scenes, so some random-arm val scenes
contributed to those two numbers per channel. That leaks two scalars per
channel, in the direction of the effect being measured, and is negligible next
to sharing regions.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/dl-local/ai4arctic/cache_x5")
    ap.add_argument("--out", default="configs/split_random_v1.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = json.loads((Path(os.path.expanduser(args.cache)) / "index.json").read_text())
    scenes = index["scenes"]
    test = [s for s in scenes if s["split"] == "test"]
    pool = [s for s in scenes if s["split"] != "test"]
    n_val = sum(s["split"] == "val" for s in scenes)          # same val size as region arm

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    val = [pool[i] for i in order[:n_val]]
    train = [pool[i] for i in order[n_val:]]

    split = {s["name"]: "test" for s in test}
    split.update({s["name"]: "val" for s in val})
    split.update({s["name"]: "train" for s in train})

    train_regions = {s["region"] for s in train}
    shared = sum(s["region"] in train_regions for s in val)
    out = {
        "version": "random_v1", "unit": "scene", "seed": args.seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "val_scenes_whose_region_is_in_train": shared,
        "scenes": split,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"random scene split (seed {args.seed}): {out['counts']}")
    print(f"val scenes whose region also appears in train: {shared}/{len(val)}"
          "   (region arm: 0/24)")
    print("val regions:", dict(sorted(Counter(s["region"] for s in val).items())))
    print("test regions (unchanged):", sorted({s["region"] for s in test}))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
