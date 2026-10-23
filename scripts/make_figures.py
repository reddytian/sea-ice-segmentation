#!/usr/bin/env python3
"""Figures for the README, built only from files the runs already wrote
(log.jsonl, *_report.json, diagnose_val.json). No model inference.

    docs/figures/metric_vs_floor.png     test: AutoICE combined vs domain-aware sel90, each next to its floor
    docs/figures/selection_bias.png      val sel90 per epoch, plateau band, selected epoch, test score
    docs/figures/amsr_ablation.png       plateau val scores, 17 channels vs SAR only
    docs/figures/val_vs_test_classes.png SOD/FLOE ice-class IoU (SIC >= 90%) on val vs test
    docs/figures/ice_to_water.png        SOD ice predicted as water, by true SIC (needs diagnose_val.json)
    docs/figures/leakage_b.png           experiment B: model A vs B on the same eval scenes (needs runs/leak_*)

Figures whose inputs are missing are skipped with a note.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated two-slot categorical palette (blue, orange) on the light surface; the
# floor is a reference, not a series, so it is neutral gray with a hatch.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, FLOOR = "#2a78d6", "#eb6834", "#b9b8b3"
SOD_NAMES = ["new", "young", "thin FY", "thick FY", "old"]
FLOE_NAMES = ["cake", "small", "medium", "big", "vast"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def load_log(run: Path):
    return [r for r in map(json.loads, (run / "log.jsonl").read_text().splitlines()) if "epoch" in r]


def plateau(run: Path, get, last=10):
    v = np.array([get(r["val"]) for r in load_log(run)[-last:]], dtype=float)
    return v.mean(), v.std(ddof=1)


def bar_labels(ax, bars, fmt="{:.2f}", err=None):
    for k, b in enumerate(bars):
        h = b.get_height()
        top = h + (err[k] if err is not None else 0)
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, top), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)


GAP = 0.02   # surface gap between the two bars of a pair (axis units)


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print(f"  wrote {out / name}")


def fig_metric_vs_floor(runs: Path, out: Path):
    rep = json.loads((runs / "unet_r2" / "test_report.json").read_text())
    names = ["POOLED"] + list(rep["regions"])
    rows = [(rep["pooled"], rep["pooled_floor"])] + [(s, s["floor"]) for s in rep["regions"].values()]
    labels = ["all test" if n == "POOLED" else n.replace("SGRDI", "CIS ") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x, w = np.arange(len(names)), 0.34
    for ax, key, title in [(axes[0], "combined", "AutoICE combined score"),
                           (axes[1], "sel90", "Domain-aware score (sel90)")]:
        m = [r[0][key] for r in rows]
        f = [r[1][key] for r in rows]
        b1 = ax.bar(x - w / 2 - GAP / 2, m, w, color=BLUE, label="model")
        b2 = ax.bar(x + w / 2 + GAP / 2, f, w, color=SURFACE, edgecolor=FLOOR, hatch="////", linewidth=1,
                    label="floor: perfect water mask,\none class for all ice")
        bar_labels(ax, b1); bar_labels(ax, b2)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.0)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Test regions: the AutoICE score barely separates the model from a trivial floor",
                 x=0.01, ha="left", fontsize=11)
    save(fig, out, "metric_vs_floor.png")


def fig_selection_bias(runs: Path, out: Path):
    log = load_log(runs / "unet_r2")
    ep = np.array([r["epoch"] for r in log])
    s = np.array([r["val"]["sel90"] for r in log])
    test = json.loads((runs / "unet_r2" / "test_report.json").read_text())["pooled"]["sel90"]
    mu, sd = s[-10:].mean(), s[-10:].std(ddof=1)
    i = int(s.argmax())
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axhspan(mu - sd, mu + sd, xmin=(ep[-10] - ep[0]) / (ep[-1] - ep[0] + 1), color=BLUE, alpha=0.12, lw=0)
    ax.plot(ep, s, color=BLUE, lw=2, label="val sel90 per epoch")
    ax.plot(ep[i], s[i], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
    ax.axhline(test, color=ORANGE, lw=2, ls="--", label="test sel90 (selected checkpoint)")
    ax.annotate(f"selected: {s[i]:.3f} (epoch {ep[i]})", (ep[i], s[i]), xytext=(-170, 16),
                textcoords="offset points", fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(f"last-10 plateau {mu:.3f} ± {sd:.3f}", (ep[-10], mu + sd), xytext=(0, 22),
                textcoords="offset points", fontsize=9, color=INK2)
    ax.annotate(f"test {test:.3f}", (ep[-1], test), xytext=(0, -14), textcoords="offset points",
                ha="right", fontsize=9, color=INK2)
    ax.set_xlabel("epoch"); ax.set_ylabel("sel90")
    ax.set_ylim(s.min() - 0.03, s.max() + 0.07)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Picking the best of 40 noisy epochs inflates the val score; test is lower again",
                 loc="left")
    save(fig, out, "selection_bias.png")


def fig_amsr(runs: Path, out: Path):
    metrics = [("SIC R² (ice px)", lambda v: v["r2_SIC_ice"]),
               ("SOD@90 mIoU", lambda v: v["miou90"]["SOD"]),
               ("FLOE@90 mIoU", lambda v: v["miou90"]["FLOE"]),
               ("sel90", lambda v: v["sel90"]),
               ("AutoICE combined", lambda v: v["combined"])]
    full = [plateau(runs / "unet_r2", g) for _, g in metrics]
    sar = [plateau(runs / "unet_r3_sar", g) for _, g in metrics]
    x, w = np.arange(len(metrics)), 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    b1 = ax.bar(x - w / 2 - GAP / 2, [m for m, _ in full], w, yerr=[s for _, s in full], color=BLUE,
                ecolor=INK2, capsize=3, label="SAR + AMSR2 (17 channels)")
    b2 = ax.bar(x + w / 2 + GAP / 2, [m for m, _ in sar], w, yerr=[s for _, s in sar], color=ORANGE,
                ecolor=INK2, capsize=3, label="SAR only (HH, HV, incidence)")
    bar_labels(ax, b1, err=[s for _, s in full]); bar_labels(ax, b2, err=[s for _, s in sar])
    ax.set_xticks(x, [m for m, _ in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("val, mean ± sd of last 10 epochs")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Removing passive microwave halves the domain-aware score; AutoICE combined drops 0.04",
                 loc="left")
    save(fig, out, "amsr_ablation.png")


def fig_val_vs_test(runs: Path, out: Path):
    val = json.loads((runs / "unet_r2" / "val_report.json").read_text())["pooled"]["per_class_iou"]
    test = json.loads((runs / "unet_r2" / "test_report.json").read_text())["pooled"]["per_class_iou"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [5, 3]})
    for ax, key, names, idx in [(axes[0], "SOD@90", SOD_NAMES, range(1, 6)),
                                (axes[1], "FLOE@90", FLOE_NAMES, range(3, 6))]:
        v = [val[key][k] for k in idx]
        t = [test[key][k] for k in idx]
        keep = [j for j in range(len(v)) if v[j] is not None and t[j] is not None]
        x, w = np.arange(len(keep)), 0.34
        b1 = ax.bar(x - w / 2 - GAP / 2, [v[j] for j in keep], w, color=BLUE, label="val regions")
        b2 = ax.bar(x + w / 2 + GAP / 2, [t[j] for j in keep], w, color=ORANGE, label="test regions")
        bar_labels(ax, b1); bar_labels(ax, b2)
        ax.set_xticks(x, [list(names)[k - (1 if key == "SOD@90" else 1)] for k in [list(idx)[j] for j in keep]])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{key.split('@')[0]} IoU where true SIC ≥ 90%", loc="left")
    axes[0].set_ylabel("IoU (selected checkpoint)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Same model, different regions: young ice is found on val and lost on test",
                 x=0.01, ha="left", fontsize=11)
    save(fig, out, "val_vs_test_classes.png")


def fig_ice_to_water(runs: Path, out: Path):
    series = [(r, json.loads((runs / r / "diagnose_val.json").read_text()))
              for r in ("unet_base32", "unet_r2") if (runs / r / "diagnose_val.json").exists()]
    if not series:
        print("  skip ice_to_water.png (run scripts/diagnose_val.py first)")
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    sic = np.arange(10, 101, 10)
    ax.plot(sic, 100 - sic, color=INK2, lw=1.5, ls=":", label="open-water share inside the polygon (100 − SIC)")
    for (name, d), color in zip(series, (ORANGE, BLUE)):
        rows = {r["sic_pct"]: r["pred_water_pct"] for r in d["ice_to_water_by_sic"]}
        ax.plot(sic, [rows.get(int(s), np.nan) for s in sic], color=color, lw=2, marker="o", ms=5,
                label={"unet_base32": "model, round 1", "unet_r2": "model, round 2 (final)"}.get(name, name))
    ax.set_xlabel("true SIC of the pixel's ice-chart polygon (%)")
    ax.set_ylabel("SOD ice pixels predicted as water (%)")
    ax.set_xticks(sic); ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("'Ice predicted as water' lives in low-concentration polygons, below the physical share",
                 loc="left")
    save(fig, out, "ice_to_water.png")


def fig_leakage_b(runs: Path, out: Path):
    a, b = runs / "leak_A", runs / "leak_B"
    if not ((a / "log.jsonl").exists() and (b / "log.jsonl").exists()) or len(load_log(b)) < 10:
        print("  skip leakage_b.png (needs finished runs/leak_A and runs/leak_B)")
        return
    metrics = [("sel90", lambda v: v["sel90"]), ("SIC R² (ice px)", lambda v: v["r2_SIC_ice"]),
               ("SOD@90 mIoU", lambda v: v["miou90"]["SOD"]), ("FLOE@90 mIoU", lambda v: v["miou90"]["FLOE"]),
               ("AutoICE combined", lambda v: v["combined"])]
    pa = [plateau(a, g) for _, g in metrics]
    pb = [plateau(b, g) for _, g in metrics]
    x, w = np.arange(len(metrics)), 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    b1 = ax.bar(x - w / 2 - GAP / 2, [m for m, _ in pa], w, yerr=[s for _, s in pa], color=BLUE, ecolor=INK2,
                capsize=3, label="A: never saw these regions")
    b2 = ax.bar(x + w / 2 + GAP / 2, [m for m, _ in pb], w, yerr=[s for _, s in pb], color=ORANGE, ecolor=INK2,
                capsize=3, label="B: saw the same regions on other dates")
    bar_labels(ax, b1, err=[s for _, s in pa]); bar_labels(ax, b2, err=[s for _, s in pb])
    ax.set_xticks(x, [m for m, _ in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("12 eval scenes, mean ± sd of last 10 epochs")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Leakage B: same eval scenes; B also lost half its young-ice training pixels", loc="left")
    save(fig, out, "leakage_b.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_metric_vs_floor, fig_selection_bias, fig_amsr, fig_val_vs_test,
              fig_ice_to_water, fig_leakage_b):
        try:
            f(runs, out)
        except FileNotFoundError as e:
            print(f"  skip {f.__name__}: missing {Path(e.filename).relative_to(Path.cwd()) if e.filename and Path(e.filename).is_absolute() and str(e.filename).startswith(str(Path.cwd())) else e.filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
