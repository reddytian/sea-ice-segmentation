"""Confusion-matrix metrics, kept per scene so any grouping can be scored later.

Same idea as dl-learning Day 11's evaluate(): accumulate a confusion matrix over
non-fill pixels, derive everything from it. Extended to three tasks and to one
matrix per scene, so the held-out split can be reported per region.

AutoICE-style scores (reported for comparability with the literature):
  r2_SIC     R^2 with class index 0..10 as concentration 0..100%
  f1w_SOD    support-weighted F1
  f1w_FLOE   support-weighted F1
  combined = (2*r2_SIC + 2*f1w_SOD + f1w_FLOE) / 5
  miou_sel   mean of the three task mIoUs (round-1 selection score)

Why those are not enough (measured on round-1 val):
  * Open water is 54-67% of valid pixels; a perfect water/ice mask with ONE
    class for all ice scores combined 0.81. `floor()` reports that model.
  * SIC is ordinal: predicting 80% for 90% is a full miss in IoU.
  * Ice-chart SOD/FLOE are one label per polygon. In a 20% concentration polygon
    ~80% of pixels are open water labelled with an ice type; 95% of the
    "ice -> water" SOD errors came from polygons at <= 50% concentration, 0% at 100%.

Domain-aware scores (round-2 selection):
  r2_SIC_ice  R^2 over pixels whose TRUE SIC > 0 (ordinal, not water-dominated)
  miou90      SOD / FLOE mIoU over ice classes, only where true SIC >= 90%
              (requires the "SOD@90" / "FLOE@90" banks filled by engine.evaluate)
  sel90       mean(r2_SIC_ice, miou90 SOD, miou90 FLOE)
"""
from __future__ import annotations

import numpy as np
import torch

FILL = 255
REPORT_ONLY = {"FLOE": [6], "FLOE@90": [6]}   # icebergs: mostly outside train
MAIN = ("SIC", "SOD", "FLOE")
STRATIFIED = {"SOD@90": 6, "FLOE@90": 7}       # filled only where true SIC >= 9


class ConfusionBank:
    def __init__(self, n_scenes: int, tasks: dict[str, int]):
        self.tasks = tasks
        self.cm = {t: np.zeros((n_scenes, n, n), dtype=np.int64) for t, n in tasks.items()}

    @torch.no_grad()
    def update(self, task: str, scene: torch.Tensor, y: torch.Tensor, pred: torch.Tensor):
        """scene: (B,) scene indices; y, pred: (B, H, W). Fill pixels are skipped."""
        n = self.tasks[task]
        y, pred = y.cpu(), pred.cpu()
        s = scene.cpu().view(-1, 1, 1).expand_as(y)
        keep = y != FILL
        idx = (s[keep] * n * n + y[keep] * n + pred[keep]).numpy()
        cm = self.cm[task]
        cm += np.bincount(idx, minlength=cm.size).reshape(cm.shape)

    def pooled(self, scenes=None) -> dict[str, np.ndarray]:
        sel = slice(None) if scenes is None else list(scenes)
        return {t: m[sel].sum(0) for t, m in self.cm.items()}

    def state_dict(self):
        return {t: m.copy() for t, m in self.cm.items()}


def iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    out = np.full(len(tp), np.nan)
    present = cm.sum(1) > 0
    out[present] = tp[present] / np.maximum(denom[present], 1)
    return out


def f1_weighted(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    f1 = 2 * tp / np.maximum(support + cm.sum(0), 1)
    return float((f1 * support).sum() / max(support.sum(), 1))


def _r2(cm: np.ndarray, rows: np.ndarray) -> float:
    """R^2 over the given true-class rows; predictions keep their full range."""
    k = np.arange(cm.shape[0], dtype=float)
    sub = cm[rows]
    kr = k[rows]
    n_true = sub.sum(1)
    N = n_true.sum()
    if N == 0:
        return float("nan")
    mean = (n_true * kr).sum() / N
    ss_tot = (n_true * (kr - mean) ** 2).sum()
    ss_res = (sub * (kr[:, None] - k[None, :]) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def r2_ordinal(cm: np.ndarray) -> float:
    return _r2(cm, np.arange(cm.shape[0]))


def r2_ice(cm: np.ndarray) -> float:
    """R^2 over pixels whose true SIC > 0. A water prediction still counts as 0%."""
    return _r2(cm, np.arange(1, cm.shape[0]))


def ice_miou(cm: np.ndarray, drop=()) -> float:
    """mIoU over ice classes (1..n-1) present in truth, minus report-only classes."""
    v = iou(cm)
    v = np.array([x for c, x in enumerate(v) if c >= 1 and c not in drop])
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _rnd(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def summarize(cms: dict[str, np.ndarray]) -> dict:
    out = {"per_class_iou": {}, "miou": {}}
    for t, cm in cms.items():
        v = iou(cm)
        out["per_class_iou"][t] = [None if np.isnan(x) else round(float(x), 4) for x in v]
        out["miou"][t] = _rnd(np.nanmean(v)) if np.isfinite(v).any() else None
    out["r2_SIC"] = _rnd(r2_ordinal(cms["SIC"]))
    out["f1w_SOD"] = _rnd(f1_weighted(cms["SOD"]))
    out["f1w_FLOE"] = _rnd(f1_weighted(cms["FLOE"]))
    out["combined"] = _rnd((2 * out["r2_SIC"] + 2 * out["f1w_SOD"] + out["f1w_FLOE"]) / 5)
    sel = []
    for t in MAIN:
        v = iou(cms[t])
        drop = REPORT_ONLY.get(t, [])
        v = np.array([x for c, x in enumerate(v) if c not in drop])
        if np.isfinite(v).any():
            sel.append(np.nanmean(v))
    out["miou_sel"] = _rnd(np.mean(sel))

    out["r2_SIC_ice"] = _rnd(r2_ice(cms["SIC"]))
    if all(k in cms for k in STRATIFIED):
        out["miou90"] = {t: _rnd(ice_miou(cms[f"{t}@90"], REPORT_ONLY.get(f"{t}@90", [])))
                         for t in ("SOD", "FLOE")}
        parts = [r2_ice(cms["SIC"]), ice_miou(cms["SOD@90"]),
                 ice_miou(cms["FLOE@90"], REPORT_ONLY["FLOE@90"])]
        out["sel90"] = _rnd(np.mean(parts)) if np.all(np.isfinite(parts)) else None
    return out


def floor(cms: dict[str, np.ndarray]) -> dict:
    """Scores of a model with a PERFECT water/ice mask that predicts one constant
    class for every ice pixel -- for each score, the constant that maximises it.

    Needs only the true label counts (row sums), so it can be computed for any
    split or region from the real model's confusion matrices.
    """
    def const(hist, c):
        n = len(hist)
        m = np.zeros((n, n), dtype=np.int64)
        m[0, 0] = hist[0]
        m[1:, c] = hist[1:]
        return m

    def best(t, score):
        hist = cms[t].sum(1)
        return max((const(hist, c) for c in range(1, len(hist))), key=score)

    fake = {"SIC": best("SIC", r2_ordinal),
            "SOD": best("SOD", f1_weighted),
            "FLOE": best("FLOE", f1_weighted)}
    for t in STRATIFIED:
        if t in cms:
            drop = REPORT_ONLY.get(t, [])
            fake[t] = best(t, lambda m, d=drop: np.nan_to_num(ice_miou(m, d), nan=-1))
    out = summarize(fake)
    # The constant that maximises overall R^2 is not the one that maximises ice-only R^2.
    out["r2_SIC_ice"] = _rnd(r2_ice(best("SIC", lambda m: np.nan_to_num(r2_ice(m), nan=-9))))
    if "sel90" in out:
        parts = [out["r2_SIC_ice"], out["miou90"]["SOD"], out["miou90"]["FLOE"]]
        # undefined (None) when, e.g., all ice has one concentration class
        out["sel90"] = None if None in parts else _rnd(np.mean(parts))
    return out
