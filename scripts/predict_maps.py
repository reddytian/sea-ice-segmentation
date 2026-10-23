#!/usr/bin/env python3
"""Side-by-side maps: SAR, ice chart, model -- for one scene per test region.

The model is the checkpoint already selected on val and already scored on test;
this script only draws. Which scenes are drawn is decided from the LABELS, before
any prediction is made: per region, the scene with the most pixels that have an
SOD label at >= 90% concentration (the pixels the domain-aware score uses).

Columns: SAR HH | SIC chart | SIC model | SOD chart | SOD model.
Model output is masked where the chart has no label, so the two panels of a pair
cover the same pixels. Arrays are in radar geometry (not map-projected), 400 m pixels.

    python scripts/predict_maps.py --run-dir runs/unet_r2
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.data import EvalTiles  # noqa: E402
from seaice.engine import pick_device  # noqa: E402
from seaice.model import MultiTaskUNet  # noqa: E402

FILL = 255
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
NO_LABEL = "#e1e0d9"
# SIC 0..100%: sequential blue ramp, steps 100..600 of the reference palette.
SIC_COLORS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
# SOD: open water = SIC 0% blue; ice stages new -> old on a validated one-hue
# orange ordinal ramp (light = young/thin, dark = old/thick).
SOD_COLORS = ["#cde2fb", "#ff8e64", "#ea6b3a", "#cd501a", "#aa3b01", "#842c00"]
SOD_NAMES = ["open water", "new", "young", "thin first-year", "thick first-year", "old"]


def cmap_for(colors):
    cm = ListedColormap(colors + [NO_LABEL])
    n = len(colors)
    return cm, BoundaryNorm(np.arange(-0.5, n + 1), cm.N), n


def show(ax, arr, colors):
    cm, norm, n = cmap_for(colors)
    a = arr.astype(np.int16)
    a[arr == FILL] = n                     # last colour = no chart label
    ax.imshow(a, cmap=cm, norm=norm, interpolation="nearest")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/unet_r2")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="docs/figures/prediction_maps.png")
    args = ap.parse_args()

    run = Path(args.run_dir)
    device = pick_device()
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    model = MultiTaskUNet(in_ch=cfg["in_ch"], tasks=cfg["tasks"], base=cfg["base"]).to(device).eval()
    model.load_state_dict(ck["model"])
    li = cfg["label_index"]

    ds = EvalTiles(cfg["cache"], args.split, tile=cfg["eval_tile"],
                   split_map=cfg.get("split_map"), channels=cfg.get("channel_idx"))
    scenes = ds.store.scenes
    all_channels = ds.store.index["channels"]
    hh = all_channels.index("nersc_sar_primary")

    # ---- choose scenes from labels only ----
    best = {}
    for i, s in enumerate(scenes):
        _, y = ds.store.arrays(i)
        sic, sod = np.asarray(y[li["SIC"]]), np.asarray(y[li["SOD"]])
        n = int(((sic >= 9) & (sic != FILL) & (sod != FILL)).sum())
        if n > best.get(s["region"], (-1, None))[0]:
            best[s["region"]] = (n, i)
    chosen = [best[r][1] for r in sorted(best)]
    print("scenes (most SOD-labelled pixels at SIC >= 90%, per region):")
    for i in chosen:
        print(f"  {scenes[i]['region']:12s} {scenes[i]['name'][:15]}  {best[scenes[i]['region']][0]:,} px")

    # ---- predict the chosen scenes, tile by tile ----
    pred = {i: {t: np.full(scenes[i]["shape"], FILL, np.uint8) for t in ("SIC", "SOD")} for i in chosen}
    for k, (i, r0, c0) in enumerate(ds.items):
        if i not in pred:
            continue
        x, _, _, _, _ = ds[k]
        out = model(x[None].to(device))
        H, W = scenes[i]["shape"]
        h, w = min(ds.tile, H - r0), min(ds.tile, W - c0)
        for t in ("SIC", "SOD"):
            pred[i][t][r0:r0 + h, c0:c0 + w] = out[t].argmax(1)[0, :h, :w].cpu().numpy()

    # ---- draw ----
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "font.size": 10})
    cols = ["SAR HH (dB, normalised)", "SIC: ice chart", "SIC: model", "SOD: ice chart", "SOD: model"]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(14, 2.85 * len(chosen) + 0.9), squeeze=False)
    for row, i in enumerate(chosen):
        x, y = ds.store.arrays(i)
        sar = np.asarray(x[hh], dtype=np.float32)
        lo, hi = np.percentile(sar, [2, 98])
        sic_t, sod_t = np.asarray(y[li["SIC"]]), np.asarray(y[li["SOD"]])
        sic_p = np.where(sic_t == FILL, FILL, pred[i]["SIC"])
        sod_p = np.where(sod_t == FILL, FILL, pred[i]["SOD"])
        ax = axes[row]
        ax[0].imshow(sar, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        show(ax[1], sic_t, SIC_COLORS); show(ax[2], sic_p, SIC_COLORS)
        show(ax[3], sod_t, SOD_COLORS); show(ax[4], sod_p, SOD_COLORS)
        s = scenes[i]
        d = s["name"][:8]
        ax[0].set_ylabel(f"{s['region'].replace('SGRDI', 'CIS ')}\n{d[:4]}-{d[4:6]}-{d[6:]}",
                         fontsize=10, color=INK2)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        if row == 0:
            for a, c in zip(ax, cols):
                a.set_title(c, fontsize=10, color=INK2)

    fig.suptitle(f"{args.split.capitalize()} scenes, one per region (chosen from labels before predicting); "
                 "radar geometry, 400 m pixels", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97), h_pad=1.2)

    def under(c0, c1):                    # x-centre beneath columns c0..c1
        bottom = axes[-1]
        return (bottom[c0].get_position().x0 + bottom[c1].get_position().x1) / 2

    sic_handles = [Patch(color=SIC_COLORS[k], label=f"{k * 10}%") for k in (0, 2, 4, 6, 8, 10)]
    sod_handles = [Patch(color=c, label=n) for c, n in zip(SOD_COLORS, SOD_NAMES)]
    nolab = Patch(color=NO_LABEL, label="no chart label")
    fig.legend(handles=sic_handles + [nolab], loc="lower center", bbox_to_anchor=(under(1, 2), 0.0),
               ncol=4, frameon=False, fontsize=9, title="SIC", title_fontsize=9)
    fig.legend(handles=sod_handles, loc="lower center", bbox_to_anchor=(under(3, 4), 0.0),
               ncol=3, frameon=False, fontsize=9, title="SOD (stage of development)", title_fontsize=9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.15)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
