#!/usr/bin/env python3
"""Choose a region-held-out train/val/test split from corpus_stats.npz.

Why regions, not scenes: scenes from one region share an ice regime, an ice
service's charting conventions and, often, the same ice days apart. A random
scene split puts every region on both sides and rewards memorising regions.

Why search instead of picking regions by hand: several classes live in a few
regions (Qaanaaq alone holds ~47% of SOD 3, one SGRDIEA scene ~35% of SOD 2).
A hand-picked split can starve train of a class or leave test with no pixels
of it. The search below only looks at LABEL statistics, never at model
scores, so it is stratification, not tuning on the test set.

Constraints for each held-out split (val and test):
  * 3 regions, 19-24 scenes, at least one DMI and one CIS region
    (the two services chart differently; both must be represented)
  * every class of SIC, SOD, FLOE present in >= 2 scenes with >= MINPX pixels
  * train keeps >= 50% of every class
Objective: minimise the worst over-/under-representation of any class
(class share in split / valid-pixel share in split), then the mean.

FLOE 6 (icebergs) is report-only: 0.17% of FLOE pixels, DMI-only, and with it
as a hard constraint no partition is feasible.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

LAB = {"SIC": 11, "SOD": 6, "FLOE": 7}
MINPX = 50_000                  # roughly one 224x224 patch
REPORT_ONLY = {("FLOE", 6)}
N_HELD = 3
SCENE_RANGE = (19, 24)
TRAIN_MIN_SHARE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="~/dl-local/ai4arctic/corpus_stats.npz")
    ap.add_argument("--out", default="configs/split_region_v1.json")
    args = ap.parse_args()

    z = np.load(os.path.expanduser(args.stats), allow_pickle=False)
    reg, src = z["region"], z["source"]
    R = sorted(set(reg.tolist()))
    svc = {r: str(src[reg == r][0]) for r in R}
    H = {l: z["hist_" + l][:, :n] for l, n in LAB.items()}

    px = {l: np.array([H[l][reg == r].sum(0) for r in R]) for l in LAB}
    nsc = {l: np.array([(H[l][reg == r] >= MINPX).sum(0) for r in R]) for l in LAB}
    nscene = np.array([(reg == r).sum() for r in R])
    tot = {l: px[l].sum(0) for l in LAB}
    keep = {l: np.array([(l, c) not in REPORT_ONLY for c in range(n)])
            for l, n in LAB.items()}

    def shares(idx):
        idx = list(idx)
        out = {}
        for l in LAB:
            p = px[l][idx].sum(0)
            out[l] = (p / tot[l], p.sum() / tot[l].sum(), nsc[l][idx].sum(0))
        return out

    ids = range(len(R))
    found = []
    for a in itertools.combinations(ids, N_HELD):
        rest = [i for i in ids if i not in a]
        for b in itertools.combinations(rest, N_HELD):
            tr = [i for i in rest if i not in b]
            ok, devs = True, []
            for part in (a, b):
                if len({svc[R[i]] for i in part}) < 2:
                    ok = False
                    break
                if not SCENE_RANGE[0] <= nscene[list(part)].sum() <= SCENE_RANGE[1]:
                    ok = False
                    break
                for l, (s, vs, n) in shares(part).items():
                    if (n[keep[l]] < 2).any():
                        ok = False
                        break
                    devs += list(np.abs(np.log(np.clip(s / vs, 1e-3, None)))[keep[l]])
                if not ok:
                    break
            if not ok:
                continue
            if any((s[keep[l]] < TRAIN_MIN_SHARE).any()
                   for l, (s, _, _) in shares(tr).items()):
                continue
            found.append((max(devs), float(np.mean(devs)), a, b))

    if not found:
        raise SystemExit("no feasible partition")
    found.sort(key=lambda x: (x[0], x[1]))
    worst, mean, a, b = found[0]
    print(f"{len(found)} feasible partitions; best worst-case {np.exp(worst):.2f}x")

    # The best result comes as a mirror pair (a, b) / (b, a). Test gets the side
    # with fewer near-empty scenes: test is the reported number, so it should be
    # the more evaluable half.
    fill = z["hist_SIC"][:, 255] / z["hist_SIC"].sum(1)

    def n_empty(part):
        return int((np.isin(reg, [R[i] for i in part]) & (fill > 0.9)).sum())

    test, val = (a, b) if n_empty(a) <= n_empty(b) else (b, a)
    train = [i for i in ids if i not in test and i not in val]

    split = {
        "version": "region_v1",
        "unit": "region",
        "test": sorted(R[i] for i in test),
        "val": sorted(R[i] for i in val),
        "train": sorted(R[i] for i in train),
        "expected_scenes": {
            "train": int(nscene[train].sum()),
            "val": int(nscene[list(val)].sum()),
            "test": int(nscene[list(test)].sum()),
        },
        "report_only_classes": {"FLOE": [6]},
        "selection": {
            "min_px_per_scene": MINPX,
            "worst_class_ratio": round(float(np.exp(worst)), 3),
            "feasible_partitions": len(found),
            "uses_model_scores": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps({k: split[k] for k in ("test", "val", "expected_scenes")}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
