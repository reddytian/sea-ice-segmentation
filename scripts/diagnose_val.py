#!/usr/bin/env python3
"""Val-only diagnostics that need pixel-level joint labels, not just confusion matrices.

  A. per scene: SIC water->ice and ice->water error rates
     (is SouthWest's below-floor score wind-roughened water called ice?)
  B. SOD ice pixels predicted as WATER, broken down by the TRUE SIC of the pixel
     (ice-chart SOD is one label per polygon; in a 30% concentration polygon most
      pixels are open water between floes but are labelled with the ice type.
      If the "ice -> water" errors sit at low true SIC, they are label resolution,
      not model failure.)
  C. SOD per-class recall on all pixels vs near-full-concentration pixels (SIC >= 9)
  D. thin first-year ice (SOD 3) predicted as old ice (SOD 5), by region

Refuses --split test: diagnosing on test would turn it into a second val set.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seaice.data import EvalTiles  # noqa: E402
from seaice.engine import pick_device  # noqa: E402
from seaice.model import MultiTaskUNet  # noqa: E402

FILL = 255
SOD_NAMES = ["water", "new", "young", "thinFY", "thickFY", "old"]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/unet_base32")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run_dir)
    device = pick_device()
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    model = MultiTaskUNet(in_ch=cfg["in_ch"], tasks=cfg["tasks"], base=cfg["base"]).to(device).eval()
    model.load_state_dict(ck["model"])
    li = cfg["label_index"]

    ds = EvalTiles(cfg["cache"], "val", tile=cfg["eval_tile"],
                   split_map=cfg.get("split_map"), channels=cfg.get("channel_idx"))
    S = len(ds.store.scenes)
    sic_cm = np.zeros((S, 11, 11), dtype=np.int64)            # scene, true SIC, pred SIC
    joint = np.zeros((S, 11, 6, 6), dtype=np.int64)          # scene, true SIC, true SOD, pred SOD

    for x, y, scene, _, _ in DataLoader(ds, batch_size=args.batch, shuffle=False):
        out = model(x.to(device))
        sic_t, sod_t = y[:, li["SIC"]], y[:, li["SOD"]]
        sic_p = out["SIC"].argmax(1).cpu()
        sod_p = out["SOD"].argmax(1).cpu()
        sc = scene.view(-1, 1, 1).expand_as(sic_t)

        k = sic_t != FILL
        idx = (sc[k] * 11 + sic_t[k]) * 11 + sic_p[k]
        sic_cm += np.bincount(idx.numpy(), minlength=sic_cm.size).reshape(sic_cm.shape)

        k = (sic_t != FILL) & (sod_t != FILL)
        idx = ((sc[k] * 11 + sic_t[k]) * 6 + sod_t[k]) * 6 + sod_p[k]
        joint += np.bincount(idx.numpy(), minlength=joint.size).reshape(joint.shape)

    scenes = ds.store.scenes

    print("A. SIC water/ice errors per scene (checkpoint epoch %d)" % ck["epoch"])
    print("   %-10s %-15s %11s %11s %10s %10s" % ("region", "scene", "water->ice", "ice->water",
                                                 "water_px", "ice_px"))
    for s, m in sorted(zip(scenes, sic_cm), key=lambda t: (t[0]["region"], t[0]["name"])):
        w, i = m[0].sum(), m[1:].sum()
        print("   %-10s %-15s %11.3f %11.3f %10d %10d" % (
            s["region"], s["name"][:15], m[0, 1:].sum() / max(w, 1), m[1:, 0].sum() / max(i, 1), w, i))

    J = joint.sum(0)                                          # true SIC, true SOD, pred SOD
    report = {"checkpoint_epoch": ck["epoch"], "ice_to_water_by_sic": [], "sod_recall": [],
              "thin_fy_by_region": {},
              "sic_errors_by_scene": [{"region": s["region"], "scene": s["name"],
                                       "water_to_ice": float(m[0, 1:].sum() / max(m[0].sum(), 1)),
                                       "ice_to_water": float(m[1:, 0].sum() / max(m[1:].sum(), 1)),
                                       "water_px": int(m[0].sum()), "ice_px": int(m[1:].sum())}
                                      for s, m in zip(scenes, sic_cm)]}
    print("\nB. SOD ice pixels (true SOD 1-5) predicted as water, by TRUE SIC")
    print("   %8s %12s %14s" % ("true SIC", "ice px", "pred water %"))
    for c in range(1, 11):
        ice = J[c, 1:, :].sum()
        if ice:
            pct = J[c, 1:, 0].sum() / ice * 100
            report["ice_to_water_by_sic"].append({"sic_pct": c * 10, "ice_px": int(ice),
                                                  "pred_water_pct": round(float(pct), 2)})
            print("   %7d0%% %12d %13.1f%%" % (c, ice, pct))

    print("\nC. SOD recall per class: all SIC-labelled pixels vs near-full concentration (SIC >= 9)")
    allc, full = J.sum(0), J[9:].sum(0)
    print("   %-8s %10s %12s %10s %12s" % ("class", "recall all", "px all", "recall >=90", "px >=90"))
    for k in range(6):
        ra = allc[k, k] / max(allc[k].sum(), 1)
        rf = full[k, k] / max(full[k].sum(), 1)
        report["sod_recall"].append({"class": SOD_NAMES[k], "recall_all": float(ra), "px_all": int(allc[k].sum()),
                                     "recall_sic90": float(rf), "px_sic90": int(full[k].sum())})
        print("   %-8s %10.3f %12d %10.3f %12d" % (SOD_NAMES[k], ra, allc[k].sum(), rf, full[k].sum()))

    print("\nD. thin first-year ice (SOD 3): where its pixels go, by region")
    by_region = defaultdict(list)
    for i, s in enumerate(scenes):
        by_region[s["region"]].append(i)
    print("   %-10s %10s   " % ("region", "thinFY px") + "  ".join("%7s" % n for n in SOD_NAMES))
    for r, ix in sorted(by_region.items()):
        row = joint[ix].sum((0, 1))[3]
        if row.sum():
            report["thin_fy_by_region"][r] = {"px": int(row.sum()),
                                              "pred_pct": {n: round(float(v / row.sum() * 100), 2)
                                                           for n, v in zip(SOD_NAMES, row)}}
            print("   %-10s %10d   " % (r, row.sum()) + "  ".join("%6.1f%%" % (v / row.sum() * 100) for v in row))
    import json
    (run / "diagnose_val.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n-> {run / 'diagnose_val.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
