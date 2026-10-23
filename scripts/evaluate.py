#!/usr/bin/env python3
"""Score a trained checkpoint on a held-out split, pooled and per region.

The test split is meant to be scored ONCE, on the checkpoint already chosen on
val. Scoring test writes <run>/test_report.json, and a second test run refuses
unless --force is given -- so peeking at test shows up as a deliberate act in
the shell history rather than a habit.

Usage:
    python scripts/evaluate.py --run-dir runs/unet_base32 --split val
    python scripts/evaluate.py --run-dir runs/unet_base32 --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.data import EvalTiles  # noqa: E402
from seaice.engine import evaluate, pick_device  # noqa: E402
from seaice.metrics import REPORT_ONLY, floor, summarize  # noqa: E402
from seaice.model import MultiTaskUNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--cache", default=None, help="defaults to the cache used in training")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ckpt", default="best.pt", help="best.pt (selected on val) or last.pt")
    args = ap.parse_args()

    run = Path(args.run_dir)
    tag = "" if args.ckpt == "best.pt" else "_" + Path(args.ckpt).stem
    out_path = run / f"{args.split}{tag}_report.json"
    if args.split == "test" and out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists: test has already been scored for this run. "
                 "Use --force only if you mean to look again.")

    device = pick_device()
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = MultiTaskUNet(in_ch=cfg["in_ch"], tasks=cfg["tasks"], base=cfg["base"]).to(device)
    model.load_state_dict(ck["model"])

    ds = EvalTiles(args.cache or cfg["cache"], args.split, tile=cfg["eval_tile"],
                   split_map=cfg.get("split_map"), channels=cfg.get("channel_idx"))
    bank, _, secs = evaluate(model, ds, device, cfg["tasks"], cfg["label_index"], args.batch)

    pooled = summarize(bank.pooled())
    by_region = defaultdict(list)
    for i, s in enumerate(ds.store.scenes):
        by_region[s["region"]].append(i)
    regions = {r: summarize(bank.pooled(ix)) | {"scenes": len(ix),
                                                 "floor": floor(bank.pooled(ix))}
               for r, ix in sorted(by_region.items())}
    pooled_floor = floor(bank.pooled())

    report = {"split": args.split, "checkpoint_epoch": ck["epoch"], "val_at_selection": ck["val"],
              "pooled": pooled, "pooled_floor": pooled_floor, "regions": regions,
              "report_only_classes": REPORT_ONLY, "seconds": round(secs, 1)}
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    np.savez_compressed(run / f"{args.split}{tag}_cm.npz", **bank.state_dict())

    print(f"{args.split.upper()}  {args.ckpt} epoch {ck['epoch']}  ({len(ds.store.scenes)} scenes, {secs:.0f}s)")
    def fmt(v, w=7, p=3):
        return f"{'-':>{w}s}" if v is None else f"{v:{w}.{p}f}"

    print("FLOOR = perfect water/ice mask + one constant class for all ice (from true labels)")
    rows = [("POOLED", pooled, pooled_floor)] + [(r, s, s["floor"]) for r, s in regions.items()]

    print("\nAutoICE-style (all labelled pixels)")
    print(f"{'':22s} {'R2 SIC':>7s} {'F1 SOD':>7s} {'F1 FLOE':>8s} {'combined':>9s} {'mIoU':>7s}"
          f"   floor: {'comb':>6s} {'mIoU':>6s}")
    for name, s, f in rows:
        n = f" ({s['scenes']})" if "scenes" in s else ""
        print(f"{(name + n):22s} {fmt(s['r2_SIC'])} {fmt(s['f1w_SOD'])} {fmt(s['f1w_FLOE'], 8)} "
              f"{fmt(s['combined'], 9, 4)} {fmt(s['miou_sel'], 7, 4)}          "
              f"{fmt(f['combined'], 6)} {fmt(f['miou_sel'], 6)}")

    print("\nDomain-aware: SIC R2 on ice pixels; SOD/FLOE ice-class mIoU where true SIC >= 90%")
    print(f"{'':22s} {'R2 ice':>7s} {'SOD@90':>7s} {'FLOE@90':>8s} {'sel90':>9s}"
          f"          floor: {'sel90':>6s}")
    for name, s, f in rows:
        n = f" ({s['scenes']})" if "scenes" in s else ""
        m90, f90 = s.get("miou90", {}), f.get("miou90", {})
        print(f"{(name + n):22s} {fmt(s['r2_SIC_ice'])} {fmt(m90.get('SOD'))} {fmt(m90.get('FLOE'), 8)} "
              f"{fmt(s.get('sel90'), 9, 4)}                 {fmt(f.get('sel90'), 6)}")
    print("\nper-class IoU (pooled):")
    for t, v in pooled["per_class_iou"].items():
        cells = []
        for c, x in enumerate(v):
            mark = "*" if c in REPORT_ONLY.get(t, []) else ""
            cells.append(f"{c}{mark}:" + ("  -  " if x is None else f"{x:.3f}"))
        print(f"  {t:7s} mIoU {fmt(pooled['miou'][t], 5)} | " + "  ".join(cells))
    print("  * report-only class (FLOE 6, icebergs: mostly outside train)")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
