#!/usr/bin/env python3
"""Export a training run as a self-contained model bundle for seaice-predict.

    python scripts/export_model.py --run-dir runs/unet_r2 --release v0.2.0

The bundle holds the weights of the checkpoint that was selected on val and scored
on test (best.pt), the architecture, the input channels in order, the train-only
normalisation from the cache index, and the preprocessing constants. It contains
only tensors and plain values, so it loads with torch.load(weights_only=True).

After writing, the bundle is reloaded and checked to give bit-identical outputs to
the original checkpoint on a random input. With --release TAG it also writes
docker/model.json: the URL the bundle will have as an asset of GitHub release TAG,
and its SHA-256. The Docker build downloads that URL and refuses any other file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.model import MultiTaskUNet  # noqa: E402
from seaice.predict import load_bundle, make_bundle, sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/unet_r2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default="dist/seaice_unet_r2.pt")
    ap.add_argument("--release", help="GitHub release tag the bundle will be attached to, e.g. v0.2.0")
    ap.add_argument("--repo", default="reddytian/sea-ice-segmentation")
    args = ap.parse_args()

    run = Path(args.run_dir)
    ck = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    index = json.loads((Path(os.path.expanduser(cfg["cache"])) / "index.json").read_text())
    idx = cfg.get("channel_idx") or list(range(len(index["channels"])))
    channels = [index["channels"][i] for i in idx]
    if channels != cfg["channels"]:
        sys.exit(f"cache index channels {channels} != run channels {cfg['channels']}")

    model = MultiTaskUNet(in_ch=cfg["in_ch"], tasks=cfg["tasks"], base=cfg["base"])
    model.load_state_dict(ck["model"])
    model.eval()

    report = run / "test_report.json"
    test = json.loads(report.read_text())["pooled"] if report.exists() else {}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source = {"run": run.name, "checkpoint": args.ckpt, "epoch": ck.get("epoch"),
              "selected_on": cfg.get("select"), "split_file": cfg.get("split_file"),
              "val_sel90": (ck.get("val") or {}).get("sel90"),
              "test_sel90": test.get("sel90"), "test_combined": test.get("combined"),
              "code_commit": commit}
    bundle = make_bundle(model, in_ch=cfg["in_ch"], base=cfg["base"], tasks=cfg["tasks"],
                         channels=channels,
                         mean=[index["norm"]["mean"][i] for i in idx],
                         std=[index["norm"]["std"][i] for i in idx],
                         factor=index["factor"], amsr_spacing=index["amsr_spacing"],
                         amsr_center=index["amsr_center"], eval_tile=cfg["eval_tile"],
                         source=source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)

    # Reload exactly as seaice-predict does and compare outputs bit for bit.
    reloaded, _ = load_bundle(out, torch.device("cpu"))
    x = torch.randn(2, cfg["in_ch"], 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = model(x), reloaded(x)
    worst = max((a[t] - b[t]).abs().max().item() for t in a)
    if worst != 0.0:
        sys.exit(f"reloaded bundle differs from checkpoint (max abs diff {worst})")

    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  sha256 {sha256(out)}")
    print(f"  run {run.name}/{args.ckpt} epoch {source['epoch']}  val sel90 {source['val_sel90']}"
          f"  test sel90 {source['test_sel90']}")
    print(f"  {len(channels)} channels: {', '.join(channels)}")
    print("  reload check: identical outputs")
    if args.release:
        spec = {"url": f"https://github.com/{args.repo}/releases/download/{args.release}/{out.name}",
                "sha256": sha256(out), "release": args.release, "run": run.name}
        Path("docker").mkdir(exist_ok=True)
        Path("docker/model.json").write_text(json.dumps(spec, indent=1) + "\n")
        print(f"  docker/model.json -> {spec['url']}")
        print(f"  next: attach {out} to GitHub release {args.release}, then commit docker/model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
