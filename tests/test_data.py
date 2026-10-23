import json
import pickle

import numpy as np
import torch

from seaice.data import EvalTiles, TrainCrops, channel_indices, class_weights

FILL = 255


def test_eval_tiles_cover_every_pixel_exactly_once(tiny_cache):
    ds = EvalTiles(tiny_cache, "train", tile=16)
    for i, s in enumerate(ds.store.scenes):
        H, W = s["shape"]
        cover = np.zeros((H, W), int)
        rebuilt = np.zeros((3, H, W), np.uint8)
        for k, (si, r0, c0) in enumerate(ds.items):
            if si != i:
                continue
            _, y, _, _, _ = ds[k]
            h, w = min(16, H - r0), min(16, W - c0)
            cover[r0:r0 + h, c0:c0 + w] += 1
            rebuilt[:, r0:r0 + h, c0:c0 + w] = y[:, :h, :w].numpy()
        assert (cover == 1).all()
        assert np.array_equal(rebuilt, ds.store.arrays(i)[1])


def test_crop_larger_than_scene_is_padded_with_fill(tiny_cache):
    ds = TrainCrops(tiny_cache, size=32, samples=200, flips=False)
    small = [s["name"] for s in ds.store.scenes].index("20190201T000000_dmi_prep")
    torch.manual_seed(0)
    seen = 0
    for _ in range(200):
        x, y = ds[0]
        if (y[:, 20:, :] == FILL).all() and (y[:, :, 20:] == FILL).all():
            seen += 1
            assert (x[:, 20:, :] == 0).all() and (x[:, :, 20:] == 0).all()
    assert seen > 0, f"the 20x20 scene (index {small}) was never drawn"


def test_flips_and_rotations_keep_inputs_and_labels_aligned(tiny_cache):
    ds = TrainCrops(tiny_cache, size=16, samples=300, flips=True, rot90=True)
    torch.manual_seed(1)
    for k in range(300):
        x, y = ds[k]
        valid = y[0] != FILL
        assert torch.equal(x[0][valid], y[0][valid].float())


def test_channel_subset_equals_slice_of_full_crop(tiny_cache):
    full = TrainCrops(tiny_cache, size=16, samples=20, flips=False)
    sub = TrainCrops(tiny_cache, size=16, samples=20, flips=False, channels=[0, 1, 2])
    for k in range(20):
        torch.manual_seed(k); xf, yf = full[k]
        torch.manual_seed(k); xs, ys = sub[k]
        assert xs.shape[0] == 3
        assert torch.equal(xf[:3], xs) and torch.equal(yf, ys)


def test_channel_indices():
    names = ["nersc_sar_primary", "nersc_sar_secondary", "sar_incidenceangle", "btemp_6_9h"]
    assert channel_indices(names, "sar") == [0, 1, 2]
    assert channel_indices(names, "all") == [0, 1, 2, 3]


def test_split_map_overrides_cache_split(tiny_cache):
    names = [s["name"] for s in json.loads((tiny_cache / "index.json").read_text())["scenes"]]
    m = dict(zip(names, ["val", "train", "train", "train", "test"]))
    assert [s["name"] for s in EvalTiles(tiny_cache, "val", 16, split_map=m).store.scenes] == [names[0]]
    tr = TrainCrops(tiny_cache, size=16, samples=4, split_map=m)
    assert [s["name"] for s in tr.store.scenes] == names[1:4]


def test_pickling_drops_open_memmaps(tiny_cache):
    ds = TrainCrops(tiny_cache, size=16, samples=4)
    ds[0]
    assert ds.store._open
    blob = pickle.dumps(ds)
    assert len(blob) < 10_000
    assert pickle.loads(blob).store._open == {}


def test_class_weights_from_train_only(tiny_cache):
    w = class_weights(tiny_cache, {"SIC": 11, "SOD": 6, "FLOE": 7})
    for t, v in w.items():
        nz = v[v > 0]
        assert torch.isclose(nz.mean(), torch.tensor(1.0))
    # a class with no train pixels gets zero weight
    w12 = class_weights(tiny_cache, {"SIC": 12})
    assert w12["SIC"][11] == 0
