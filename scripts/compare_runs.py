#!/usr/bin/env python3
"""Compare training runs fairly, and report the leakage experiment.

1. Plateau comparison (val only). The checkpoint score of a run is the MAX over
   ~40 noisy epochs, which is biased upward (round 2: max 0.472 vs plateau
   0.454 +/- 0.009 on sel90). Configurations are therefore compared on the mean
   +/- sd of the last N epochs, read from each run's log.jsonl.

2. Leakage table. For runs that have been scored on test (test_report.json):
   the val score at the selected checkpoint, the test score, and their
   difference (optimism). The max-over-epochs bias is in both arms' val scores,
   so the DIFFERENCE between arms isolates the split protocol.

Usage:
    python scripts/compare_runs.py runs/unet_r2 runs/unet_r3_sar runs/unet_rand
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    ("sel90", lambda v: v.get("sel90")),
    ("  R2 SIC (ice px)", lambda v: v.get("r2_SIC_ice")),
    ("  SOD@90 mIoU", lambda v: (v.get("miou90") or {}).get("SOD")),
    ("  FLOE@90 mIoU", lambda v: (v.get("miou90") or {}).get("FLOE")),
    ("mIoU (all classes)", lambda v: v.get("miou_sel")),
    ("combined (AutoICE)", lambda v: v.get("combined")),
    ("  R2 SIC", lambda v: v.get("r2_SIC")),
    ("  F1w SOD", lambda v: v.get("f1w_SOD")),
    ("  F1w FLOE", lambda v: v.get("f1w_FLOE")),
]


def load_epochs(run: Path):
    recs = []
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "epoch" in r:
            recs.append(r)
    return recs


def fmt(x, w=8):
    return f"{'-':>{w}s}" if x is None else f"{x:{w}.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs]
    names = [r.name for r in runs]
    W = max(18, max(len(n) for n in names) + 2)

    print(f"PLATEAU: mean +/- sd of val scores over the last {args.last} epochs\n")
    info = []
    for r in runs:
        cfg = json.loads((r / "config.json").read_text())
        ep = load_epochs(r)
        sel = cfg.get("select", "miou_sel")
        scores = [e["val"].get(sel) for e in ep]
        best_i = int(np.nanargmax([s if s is not None else -np.inf for s in scores]))
        info.append((cfg, ep, sel, best_i))
        desc = (f"{len(cfg.get('channels', []))} ch"
                + (f", split {Path(cfg['split_file']).stem}" if cfg.get("split_file") else ", region split"))
        print(f"  {r.name}: {len(ep)} epochs, {desc}, selected on {sel} "
              f"= {scores[best_i]:.4f} at epoch {ep[best_i]['epoch']}")
    print()
    print(f"{'':22s}" + "".join(f"{n:>{W}s}" for n in names))
    for label, get in METRICS:
        row = f"{label:22s}"
        for cfg, ep, _, _ in info:
            vals = [get(e["val"]) for e in ep[-args.last:]]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                row += f"{np.mean(vals):>{W - 8}.4f} ±{np.std(vals, ddof=1):.3f}"
            else:
                row += f"{'-':>{W}s}"
        print(row)

    tested = [(r, json.loads((r / "test_report.json").read_text()))
              for r in runs if (r / "test_report.json").exists()]
    if tested:
        print("\nLEAKAGE: val at the selected checkpoint vs test (same 24 test scenes)\n")
        print(f"{'':12s} {'':22s} {'val':>8s} {'test':>8s} {'val-test':>9s}   test floor")
        for r, rep in tested:
            v, t, f = rep["val_at_selection"], rep["pooled"], rep["pooled_floor"]
            for i, (label, key) in enumerate([("sel90", "sel90"), ("mIoU", "miou_sel"),
                                              ("combined", "combined")]):
                vv, tt = v.get(key), t.get(key)
                d = None if vv is None or tt is None else vv - tt
                print(f"{(r.name if i == 0 else ''):12s} {label:22s} {fmt(vv)} {fmt(tt)} "
                      f"{fmt(d, 9)}   {fmt(f.get(key), 8)}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
