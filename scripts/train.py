#!/usr/bin/env python3
"""Train the multi-task U-Net on the train split; select on val. Never touches test.

The test split is scored only by scripts/evaluate.py, once, on the checkpoint
this script selects (dl-learning Day 6: train to learn, val to choose, test once).

Writes to --run-dir:
    config.json   every argument, channel list, class weights
    log.jsonl     one line per epoch: train loss parts, val loss, val scores, timing
    best.pt       model weights at the best val `combined` score
    val_cm.npz    per-scene val confusion matrices of the best epoch

Quick pipeline check (a few minutes): --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.data import EvalTiles, TrainCrops, channel_indices, class_weights  # noqa: E402
from seaice.engine import evaluate, pick_device  # noqa: E402
from seaice.losses import MultiTaskCE  # noqa: E402
from seaice.metrics import floor, summarize  # noqa: E402
from seaice.model import TASKS, MultiTaskUNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/dl-local/ai4arctic/cache_x5")
    ap.add_argument("--run-dir", default="runs/unet_base32")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150, help="optimizer steps per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300, help="linear warmup steps")
    ap.add_argument("--weights", choices=["sqrt_inv", "inv", "none"], default="sqrt_inv")
    ap.add_argument("--patience", type=int, default=10, help="epochs without val improvement")
    ap.add_argument("--select", choices=["sel90", "miou_sel", "combined"], default="sel90",
                    help="val score that picks the checkpoint (see metrics.py for why)")
    ap.add_argument("--rot90", action="store_true", help="add random 90-degree rotations")
    ap.add_argument("--channels", choices=["all", "sar"], default="all",
                    help="sar = SAR HH/HV + incidence angle only (AMSR2 ablation)")
    ap.add_argument("--split-file", default=None,
                    help="JSON {\"scenes\": {name: split}} overriding the cache split "
                         "(scripts/make_random_split.py)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-tile", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 10 steps, check the pipeline")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps, args.warmup, args.run_dir = 2, 10, 5, args.run_dir + "_smoke"

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device()
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    index = json.loads((Path(os.path.expanduser(args.cache)) / "index.json").read_text())
    ch_idx = channel_indices(index["channels"], args.channels)
    split_map = (json.loads(Path(args.split_file).read_text())["scenes"]
                 if args.split_file else None)

    train_ds = TrainCrops(args.cache, size=args.crop, samples=args.steps * args.batch,
                          rot90=args.rot90, split_map=split_map, channels=ch_idx)
    val_ds = EvalTiles(args.cache, "val", tile=args.eval_tile, split_map=split_map, channels=ch_idx)
    labels = train_ds.store.index["labels"]
    label_index = {t: labels.index(t) for t in TASKS}
    in_ch = len(train_ds.store.channels)

    cw = (None if args.weights == "none"
          else class_weights(args.cache, TASKS, args.weights, split_map=split_map))
    criterion = MultiTaskCE(label_index, cw).to(device)
    model = MultiTaskUNet(in_ch=in_ch, tasks=TASKS, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                    persistent_workers=args.workers > 0, drop_last=True)

    config = {**vars(args), "device": str(device), "in_ch": in_ch, "params": n_params,
              "channels": train_ds.store.channels, "channel_idx": ch_idx, "split_map": split_map,
              "tasks": TASKS, "label_index": label_index,
              "class_weights": None if cw is None else {t: w.tolist() for t, w in cw.items()},
              "train_scenes": len(train_ds.store.scenes), "val_scenes": len(val_ds.store.scenes),
              "val_tiles": len(val_ds)}
    (run / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(f"device {device} | {n_params/1e6:.2f}M params | {in_ch} channels | "
          f"train {len(train_ds.store.scenes)} scenes, val {len(val_ds.store.scenes)} scenes "
          f"({len(val_ds)} tiles)")

    best, best_ep, since = -math.inf, 0, 0
    log = open(run / "log.jsonl", "a")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        # Accumulate on the device; one .item() per epoch instead of several per step.
        sums = {t: torch.zeros((), device=device) for t in TASKS}
        tot_sum = torch.zeros((), device=device)
        for step, (x, y) in enumerate(dl, 1):
            x, y = x.to(device), y.to(device)
            loss, parts = criterion(model(x), y)
            if not torch.isfinite(loss):          # the one sync per step, kept on purpose
                raise SystemExit(f"non-finite loss at epoch {ep} step {step}: "
                                 f"{ {t: v.item() for t, v in parts.items()} }")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot_sum += loss.detach()
            for t in TASKS:
                sums[t] += parts[t]
        tot_sum = tot_sum.item()
        sums = {t: v.item() for t, v in sums.items()}
        t_train = time.time() - t0

        bank, val_loss, t_val = evaluate(model, val_ds, device, TASKS, label_index,
                                         args.eval_batch, 0, criterion)
        s = summarize(bank.pooled())
        if ep == 1:
            fl = floor(bank.pooled())   # depends only on val labels, so compute once
            log.write(json.dumps({"val_floor": fl}) + "\n")
            print(f"val floor (perfect water/ice mask, one class for all ice): "
                  f"combined {fl['combined']:.4f}  mIoU_sel {fl['miou_sel']:.4f}  sel90 {fl['sel90']:.4f}")
        rec = {"epoch": ep, "lr": sched.get_last_lr()[0],
               "train_loss": tot_sum / args.steps,
               "train_parts": {t: v / args.steps for t, v in sums.items()},
               "val_loss": val_loss, "val": s, "t_train": round(t_train, 1), "t_val": round(t_val, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()

        flag = ""
        score = s[args.select] if s[args.select] is not None else -math.inf
        if score > best:
            best, best_ep, since = score, ep, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep,
                        "val": s}, run / "best.pt")
            np.savez_compressed(run / "val_cm.npz", **bank.state_dict())
            flag = "  *best"
        else:
            since += 1
        print(f"ep {ep:3d} | loss {rec['train_loss']:.3f} (val {val_loss:.3f}) | "
              f"R2 SIC {s['r2_SIC']:.3f}  F1 SOD {s['f1w_SOD']:.3f}  F1 FLOE {s['f1w_FLOE']:.3f}"
              f"  -> {s['combined']:.4f} | mIoU {s['miou_sel']:.4f} | sel90 {s['sel90']:.4f} | "
              f"{t_train:.0f}s+{t_val:.0f}s{flag}", flush=True)
        if since >= args.patience:
            print(f"no val improvement for {args.patience} epochs; stopping")
            break

    log.close()
    # The final-epoch weights, for evaluations that must not depend on selecting on val
    # (leakage experiment B scores fixed-epoch models on its eval scenes).
    torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "val": s},
               run / "last.pt")
    print(f"best val {args.select} {best:.4f} at epoch {best_ep} -> {run/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
