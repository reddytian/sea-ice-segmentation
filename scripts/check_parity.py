#!/usr/bin/env python3
"""Check that seaice-predict reproduces the evaluation pipeline on real scenes.

For each scene (default: the test split):
  1. inputs   -- preprocess the raw NetCDF the way seaice-predict does and compare
                 with the cached array the model was scored on: must be bit-identical.
  2. outputs  -- run the exported bundle through seaice-predict's tiling and the
                 original checkpoint through EvalTiles; class maps must agree.
Both runs use the same device and the same tiles per batch, so on CPU they should
agree exactly. Any difference here would mean the deployed model is not the model
whose test scores the README reports.

    python scripts/check_parity.py --model dist/seaice_unet_r2.pt [--scenes 3] [--skip 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice import preprocess  # noqa: E402
from seaice.data import EvalTiles  # noqa: E402
from seaice.model import MultiTaskUNet  # noqa: E402
from seaice.predict import load_bundle, predict_classes  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dist/seaice_unet_r2.pt")
    ap.add_argument("--run-dir", default="runs/unet_r2")
    ap.add_argument("--raw", default="~/dl-local/ai4arctic/scenes")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scenes", type=int, default=3, help="how many scenes (0 = all in split)")
    ap.add_argument("--skip", type=int, default=0, help="start after this many scenes")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    bundle_model, b = load_bundle(args.model, device)
    ck = torch.load(Path(args.run_dir) / b["source"]["checkpoint"], map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    ref_model = MultiTaskUNet(in_ch=cfg["in_ch"], tasks=cfg["tasks"], base=cfg["base"])
    ref_model.load_state_dict(ck["model"])
    ref_model.to(device).eval()

    tiles = EvalTiles(cfg["cache"], args.split, tile=cfg["eval_tile"],
                      split_map=cfg.get("split_map"), channels=cfg.get("channel_idx"))
    stop = args.skip + args.scenes if args.scenes else None
    chosen = list(enumerate(tiles.store.scenes))[args.skip:stop]
    raw = Path(os.path.expanduser(args.raw))
    pp = b["preprocess"]
    ok_all = True
    for i, s in chosen:
        t0 = time.time()
        ds = preprocess.open_scene(raw / f"{s['name']}.nc")
        try:
            x16 = preprocess.normalise(preprocess.scene_inputs(ds, b["channels"], pp["factor"],
                                                               pp["amsr_center"]),
                                       b["norm"]["mean"], b["norm"]["std"])
        finally:
            ds.close()
        cached = np.asarray(tiles.store.arrays(i)[0])
        if cfg.get("channel_idx"):
            cached = cached[cfg["channel_idx"]]
        same_in = x16.shape == cached.shape and np.array_equal(x16.view(np.uint16),
                                                               cached.view(np.uint16))

        got = predict_classes(bundle_model, x16, b["eval_tile"], device)
        H, W = s["shape"]
        ref = {t: np.full((H, W), 255, np.uint8) for t in got}
        items = [k for k, (j, _, _) in enumerate(tiles.items) if j == i]
        for k0 in range(0, len(items), 4):              # same tiles per batch as predict_classes
            batch = [tiles[k] for k in items[k0:k0 + 4]]
            out = ref_model(torch.stack([bt[0] for bt in batch]).to(device))
            for j, (_, _, _, r0, c0) in enumerate(batch):
                h, w = min(tiles.tile, H - r0), min(tiles.tile, W - c0)
                for t in ref:
                    ref[t][r0:r0 + h, c0:c0 + w] = out[t][j].argmax(0)[:h, :w].cpu().numpy()
        agree = {t: float(np.mean(got[t] == ref[t])) for t in got}
        ok = same_in and all(v == 1.0 for v in agree.values())
        ok_all &= ok
        print(f"{s['name'][:15]} {s['region']:12s} inputs {'identical' if same_in else 'DIFFER'}"
              f"  class agreement " + "  ".join(f"{t} {v:.6f}" for t, v in agree.items())
              + f"  ({time.time() - t0:.0f} s)  {'OK' if ok else 'MISMATCH'}")
    print("PARITY OK" if ok_all else "PARITY FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
