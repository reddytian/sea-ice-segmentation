#!/usr/bin/env python3
"""Leakage experiment B: same evaluation scenes, two training sets that differ
only in whether they contain OTHER DATES of the evaluation scenes' regions.

Why: experiment A (region vs random split, each scored on its own val and on
test) could not separate leakage from val-set difficulty, because the two arms
had different val scenes. Here both models are scored on the SAME scenes.

    E (eval)   the 3 region-arm val regions; per region, scenes sorted by date,
               every second one -> 4 per region, 12 scenes. Labelled "val" in
               both split files so training logs the per-epoch score on E.
    S          the other 12 scenes of those regions (interleaved in time with E)
    D          12 scenes drawn at random from the 75 region-arm train scenes
    model A    train = 75 region-arm train scenes            (never sees E's regions)
    model B    train = 75 - D + S                             (sees other dates of E's regions)

Both trained for a fixed number of epochs with no early stopping and compared
on the mean of the last 10 epochs on E -- no checkpoint is selected on E.
B - A estimates what having seen the same region on other dates is worth.

Known asymmetries, stated up front:
  * B also LOSES D (12 random scenes from other regions); if D happens to be
    unusually informative, A is favoured.
  * The cache was normalised with train-scene statistics that include D but not
    S (two scalars per channel; negligible).
  * One seed per model. Within-run epoch noise on sel90 was ~0.01-0.015.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def acq(name: str) -> datetime:
    return datetime.strptime(name[:15], "%Y%m%dT%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/dl-local/ai4arctic/cache_x5")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = json.loads((Path(os.path.expanduser(args.cache)) / "index.json").read_text())
    scenes = index["scenes"]
    test = [s["name"] for s in scenes if s["split"] == "test"]
    region_train = [s["name"] for s in scenes if s["split"] == "train"]
    val_regions = sorted({s["region"] for s in scenes if s["split"] == "val"})

    E, S = [], []
    for r in val_regions:
        rs = sorted((s["name"] for s in scenes if s["split"] == "val" and s["region"] == r), key=acq)
        S += rs[0::2]
        E += rs[1::2]

    rng = np.random.default_rng(args.seed)
    D = [region_train[i] for i in sorted(rng.choice(len(region_train), len(S), replace=False))]

    base = {n: "test" for n in test}
    base.update({n: "val" for n in E})
    map_a = {**base, **{n: "train" for n in region_train}, **{n: "unused" for n in S}}
    map_b = {**base, **{n: "train" for n in region_train if n not in D},
             **{n: "train" for n in S}, **{n: "unused" for n in D}}
    assert len(map_a) == len(map_b) == len(scenes)
    n_tr = lambda m: sum(v == "train" for v in m.values())
    assert n_tr(map_a) == n_tr(map_b)

    region = {s["name"]: s["region"] for s in scenes}
    design = {"E_eval": E, "S_same_region_other_dates": S, "D_dropped_from_B": D, "seed": args.seed}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tag, m in (("A", map_a), ("B", map_b)):
        (out / f"split_leak{tag}.json").write_text(json.dumps(
            {"version": f"leak{tag}_v1", "design": design, "scenes": m}, indent=1) + "\n")

    print(f"E: {len(E)} eval scenes from {val_regions}; train {n_tr(map_a)} scenes in both A and B")
    print(f"B swaps {len(D)} random train scenes (regions: "
          f"{sorted({region[n] for n in D})}) for {len(S)} same-region scenes")
    print("\neval scene        region        nearest same-region scene in B's train")
    for n in E:
        gaps = [abs((acq(n) - acq(m)).days) for m in S if region[m] == region[n]]
        print(f"  {n[:15]}  {region[n]:12s}  {min(gaps):4d} days")
    print(f"\n-> {out}/split_leakA.json, {out}/split_leakB.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
