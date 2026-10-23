"""Datasets over the cache written by scripts/build_cache.py.

TrainCrops  random crops for training; an "epoch" is a fixed number of crops.
EvalTiles   deterministic non-overlapping tiles that cover every pixel of every
            scene in a split exactly once, so metrics are computed on the
            whole held-out split rather than on a random sample of it.

Arrays are opened with np.load(mmap_mode="r") lazily, inside whichever process
reads them. The open handles are dropped when the dataset is pickled: macOS
DataLoader workers use spawn, and pickling an np.memmap copies the whole array.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FILL = 255


def channel_indices(names: list[str], which: str) -> list[int]:
    """'all' -> every channel; 'sar' -> SAR backscatter + incidence angle only."""
    if which == "all":
        return list(range(len(names)))
    if which == "sar":
        idx = [i for i, n in enumerate(names) if n.startswith("nersc_sar") or "incidence" in n]
        if not idx:
            raise ValueError(f"no SAR channels among {names}")
        return idx
    raise ValueError(which)


class _SceneStore:
    """Scenes of one split, optionally re-assigned by `split_map` {scene name: split},
    reading only the input channels in `channels` (indices into index['channels'])."""

    def __init__(self, cache_dir: str | os.PathLike, split: str,
                 split_map: dict[str, str] | None = None, channels: list[int] | None = None):
        self.root = Path(os.path.expanduser(str(cache_dir)))
        self.index = json.loads((self.root / "index.json").read_text())
        if split_map is None:
            self.scenes = [s for s in self.index["scenes"] if s["split"] == split]
        else:
            missing = [s["name"] for s in self.index["scenes"] if s["name"] not in split_map]
            if missing:
                raise ValueError(f"split_map lacks {len(missing)} scenes, e.g. {missing[0]}")
            self.scenes = [s for s in self.index["scenes"] if split_map[s["name"]] == split]
        if not self.scenes:
            raise ValueError(f"no scenes with split={split!r} in {self.root}")
        self.split = split
        all_names = self.index["channels"]
        self.ch = None if channels is None else list(channels)
        self.channels = all_names if self.ch is None else [all_names[i] for i in self.ch]
        self._open: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def arrays(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self._open:
            name = self.scenes[i]["name"]
            self._open[i] = (np.load(self.root / "x" / f"{name}.npy", mmap_mode="r"),
                             np.load(self.root / "y" / f"{name}.npy", mmap_mode="r"))
        return self._open[i]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open"] = {}
        return state


def _crop(x: np.ndarray, y: np.ndarray, r0: int, c0: int, size: int, ch=None):
    """Crop (C,H,W)/(3,H,W) at (r0,c0); pad past the edge with x=0, y=FILL.
    `ch`: list of input-channel indices to keep (None = all)."""
    H, W = y.shape[1:]
    r1, c1 = min(r0 + size, H), min(c0 + size, W)
    n_ch = x.shape[0] if ch is None else len(ch)
    xc = np.zeros((n_ch, size, size), dtype=np.float32)
    yc = np.full((y.shape[0], size, size), FILL, dtype=np.uint8)
    xc[:, : r1 - r0, : c1 - c0] = x[:, r0:r1, c0:c1] if ch is None else x[ch, r0:r1, c0:c1]
    yc[:, : r1 - r0, : c1 - c0] = y[:, r0:r1, c0:c1]
    return xc, yc


class TrainCrops(Dataset):
    """Random crops from the train split.

    Scenes are drawn in proportion to their number of SIC-valid pixels, so a
    scene that is 99% fill is almost never drawn. Within a scene, up to
    `tries` positions are drawn and the first with at least `min_valid` SIC-valid
    fraction is used (else the best seen). Scenes smaller than `size` are padded
    with fill, which the loss ignores.

    A crop can still have SIC labels but no SOD/FLOE labels (~35% of SIC-labelled
    ice has no SOD/FLOE label); losses.masked_ce handles that case.
    """

    def __init__(self, cache_dir, size: int = 256, samples: int = 4000,
                 min_valid: float = 0.2, tries: int = 10, flips: bool = True,
                 split: str = "train", rot90: bool = False,
                 split_map: dict[str, str] | None = None, channels: list[int] | None = None):
        self.store = _SceneStore(cache_dir, split, split_map, channels)
        self.size, self.samples = size, samples
        self.min_valid, self.tries, self.flips = min_valid, tries, flips
        self.rot90 = rot90
        w = np.array([s["shape"][0] * s["shape"][1] * s["valid_frac"]["SIC"]
                      for s in self.store.scenes], dtype=np.float64)
        self.scene_p = torch.tensor(w / w.sum())

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, _):
        # torch's RNG is seeded independently in each DataLoader worker.
        i = int(torch.multinomial(self.scene_p, 1))
        x, y = self.store.arrays(i)
        H, W = y.shape[1:]
        best, best_v = None, -1.0
        for _ in range(self.tries):
            r0 = int(torch.randint(0, max(H - self.size, 0) + 1, ()))
            c0 = int(torch.randint(0, max(W - self.size, 0) + 1, ()))
            v = float((y[0, r0:r0 + self.size, c0:c0 + self.size] != FILL).mean())
            if v > best_v:
                best, best_v = (r0, c0), v
            if v >= self.min_valid:
                break
        xc, yc = _crop(x, y, *best, self.size, self.store.ch)
        if self.flips:
            # The incidence-angle channel flips with the image, so the angle
            # geometry the model sees stays consistent.
            if torch.rand(()) < 0.5:
                xc, yc = xc[:, ::-1], yc[:, ::-1]
            if torch.rand(()) < 0.5:
                xc, yc = xc[:, :, ::-1], yc[:, :, ::-1]
        if self.rot90 and torch.rand(()) < 0.5:
            # With both flips this covers all 8 square symmetries. It swaps range and
            # azimuth, so Sentinel-1 EW HV noise stripes appear in orientations real
            # scenes never have -- an augmentation against overfitting, not physics.
            xc, yc = np.rot90(xc, 1, axes=(1, 2)), np.rot90(yc, 1, axes=(1, 2))
        return (torch.from_numpy(np.ascontiguousarray(xc)),
                torch.from_numpy(np.ascontiguousarray(yc)).long())


class EvalTiles(Dataset):
    """Non-overlapping tiles covering every pixel of every scene in `split`.

    Returns (x, y, scene_idx, r0, c0). Edge tiles are padded with fill, so each
    real pixel is scored exactly once. Use with shuffle=False.
    """

    def __init__(self, cache_dir, split: str, tile: int = 512,
                 split_map: dict[str, str] | None = None, channels: list[int] | None = None):
        self.store = _SceneStore(cache_dir, split, split_map, channels)
        self.tile = tile
        self.items = []
        for i, s in enumerate(self.store.scenes):
            H, W = s["shape"]
            for r0 in range(0, H, tile):
                for c0 in range(0, W, tile):
                    self.items.append((i, r0, c0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k):
        i, r0, c0 = self.items[k]
        x, y = self.store.arrays(i)
        xc, yc = _crop(x, y, r0, c0, self.tile, self.store.ch)
        return torch.from_numpy(xc), torch.from_numpy(yc).long(), i, r0, c0


def class_weights(cache_dir, n_classes: dict[str, int] | None = None,
                  scheme: str = "sqrt_inv", split: str = "train",
                  split_map: dict[str, str] | None = None) -> dict[str, torch.Tensor]:
    """Per-task class weights from the TRAIN split's cached labels.

    sqrt_inv: w_c ∝ 1/sqrt(freq_c), normalised to mean 1 over classes. Milder
    than plain inverse frequency, which would weight SOD 1 (~1.8% of pixels)
    about 35x open water.
    """
    n_classes = n_classes or {"SIC": 11, "SOD": 6, "FLOE": 7}
    store = _SceneStore(cache_dir, split, split_map)
    labels = store.index["labels"]
    counts = {t: np.zeros(n, dtype=np.int64) for t, n in n_classes.items()}
    for i in range(len(store.scenes)):
        _, y = store.arrays(i)
        for t, n in n_classes.items():
            counts[t] += np.bincount(np.asarray(y[labels.index(t)]).ravel(),
                                     minlength=256)[:n]
    out = {}
    for t, c in counts.items():
        f = c / c.sum()
        if scheme == "sqrt_inv":
            w = 1 / np.sqrt(np.maximum(f, 1e-8))
        elif scheme == "inv":
            w = 1 / np.maximum(f, 1e-8)
        elif scheme == "none":
            w = np.ones_like(f)
        else:
            raise ValueError(scheme)
        w[c == 0] = 0.0          # a class absent from train gets no weight
        out[t] = torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)
    return out
