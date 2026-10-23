import numpy as np
import torch

from seaice.metrics import (ConfusionBank, f1_weighted, floor, ice_miou, iou, r2_ice,
                            r2_ordinal, summarize)


def _cm(y, p, n):
    m = np.zeros((n, n), int)
    np.add.at(m, (y, p), 1)
    return m


def _r2(y, p):
    y, p = y.astype(float), p.astype(float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def test_r2_matches_direct_computation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, 5000)
    p = np.clip(y + rng.integers(-2, 3, 5000), 0, 10)
    cm = _cm(y, p, 11)
    assert np.isclose(r2_ordinal(cm), _r2(y, p))
    assert np.isclose(r2_ice(cm), _r2(y[y > 0], p[y > 0]))


def test_iou_and_weighted_f1_match_direct_computation():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 6, 3000), rng.integers(0, 6, 3000)
    cm = _cm(y, p, 6)
    for c in range(6):
        inter = ((y == c) & (p == c)).sum()
        union = ((y == c) | (p == c)).sum()
        assert np.isclose(iou(cm)[c], inter / union)
    f1 = [2 * ((y == c) & (p == c)).sum() / ((y == c).sum() + (p == c).sum()) for c in range(6)]
    support = np.bincount(y, minlength=6)
    assert np.isclose(f1_weighted(cm), (np.array(f1) * support).sum() / support.sum())
    assert np.isclose(ice_miou(cm), np.mean([iou(cm)[c] for c in range(1, 6)]))


def test_confusion_bank_skips_fill_and_keeps_scenes_apart():
    bank = ConfusionBank(2, {"SOD": 6})
    y = torch.tensor([[[0, 1], [255, 2]], [[3, 3], [4, 255]]])
    p = torch.tensor([[[0, 1], [5, 2]], [[3, 0], [4, 1]]])
    bank.update("SOD", torch.tensor([0, 1]), y, p)
    assert bank.cm["SOD"][0].sum() == 3 and bank.cm["SOD"][1].sum() == 3
    assert (bank.pooled()["SOD"] == bank.cm["SOD"].sum(0)).all()
    assert bank.cm["SOD"][1][3, 0] == 1


def test_floor_is_perfect_when_all_ice_is_one_class():
    sic = np.diag([50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30])
    sod = np.diag([50, 0, 0, 0, 0, 30])
    floe = np.diag([50, 0, 0, 0, 0, 30, 0])
    f = floor({"SIC": sic, "SOD": sod, "FLOE": floe,
               "SOD@90": np.diag([0, 0, 0, 0, 0, 30]), "FLOE@90": np.diag([0, 0, 0, 0, 0, 30, 0])})
    assert np.isclose(f["combined"], 1.0)
    assert np.isclose(f["miou90"]["SOD"], 1.0)


def test_summarize_reports_domain_aware_scores_when_stratified_banks_exist():
    rng = np.random.default_rng(2)
    cms = {t: _cm(rng.integers(0, n, 500), rng.integers(0, n, 500), n)
           for t, n in {"SIC": 11, "SOD": 6, "FLOE": 7, "SOD@90": 6, "FLOE@90": 7}.items()}
    s = summarize(cms)
    assert {"combined", "miou_sel", "r2_SIC_ice", "miou90", "sel90"} <= s.keys()
    # FLOE 6 (icebergs) is report-only: excluded from the selection score
    cms["FLOE@90"][6, 6] += 10_000
    assert summarize(cms)["miou90"]["FLOE"] == s["miou90"]["FLOE"]
