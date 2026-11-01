"""AI4Arctic scene -> model input. One implementation for the training cache
(scripts/build_cache.py) and for inference (seaice.predict), so the inputs a
deployed model sees cannot drift from the inputs it was trained and scored on.

  * SAR (80 m) is block-averaged by `factor` (5 -> 400 m).
  * Labels take the centre pixel of each block (unbiased class proportions).
  * AMSR2 (2 km grid) is resampled bilinearly with a FIXED 25-SAR-pixel spacing,
    cell k centred at SAR pixel 25k + `centre`; positions past the grid are clamped.
  * normalise() reproduces the cache exactly, including its float16 rounding:
    the reported metrics were computed on float16-rounded inputs.
"""
from __future__ import annotations

import os

import numpy as np

SAR_DIMS = ("sar_lines", "sar_samples")
AMSR_DIMS = ("2km_grid_lines", "2km_grid_samples")
SAR_REQUIRED = ("nersc_sar_primary", "nersc_sar_secondary")
LABELS = ("SIC", "SOD", "FLOE")
FILL = 255
AMSR_SPACING = 25
# distance_map dist_id 0 ("land") as stored in AI4Arctic ready-to-train files, which keep
# it standardised (ids step by ~0.0667). scripts/check_geolocation.py reports it per scene.
LAND_VALUE = -1.5762091875076294


def open_scene(path: str | os.PathLike):
    """Open a scene the way the cache was built: raw values, no CF masking/scaling."""
    import xarray as xr
    return xr.open_dataset(path, mask_and_scale=False)


def pick_channels(ds) -> tuple[list[str], list[str]]:
    """SAR HH, HV, first incidence-angle variable on the SAR grid; every btemp_* on the 2 km grid."""
    missing = [v for v in SAR_REQUIRED if v not in ds]
    if missing:
        raise ValueError(f"missing SAR variables {missing}; SAR-grid variables present: "
                         f"{[v for v in ds.data_vars if ds[v].dims == SAR_DIMS]}")
    sar = list(SAR_REQUIRED)
    inc = [v for v in ds.data_vars if "incidence" in v.lower() and ds[v].dims == SAR_DIMS]
    sar += inc[:1]
    amsr = sorted(v for v in ds.data_vars if v.startswith("btemp") and ds[v].dims == AMSR_DIMS)
    if not amsr:
        raise ValueError("no btemp_* variables on the 2 km grid; 2 km variables present: "
                         f"{[v for v in ds.data_vars if ds[v].dims == AMSR_DIMS]}")
    return sar, amsr


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float32)


def block_centre(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[f // 2:h:f, f // 2:w:f]


def amsr_to_grid(a: np.ndarray, out_hw: tuple[int, int], f: int, centre: float) -> np.ndarray:
    """Bilinear resample (C, h, w) AMSR onto the downsampled SAR grid.

    Output pixel i is centred at SAR pixel i*f + (f-1)/2; AMSR cell k at
    SAR pixel k*25 + centre. Positions outside the AMSR grid are clamped.
    """
    def axis(n_out: int, n_in: int):
        pos = (np.arange(n_out) * f + (f - 1) / 2 - centre) / AMSR_SPACING
        pos = np.clip(pos, 0, n_in - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (pos - i0).astype(np.float32)

    y0, y1, wy = axis(out_hw[0], a.shape[1])
    x0, x1, wx = axis(out_hw[1], a.shape[2])
    wy, wx = wy[None, :, None], wx[None, None, :]
    top = a[:, y0][:, :, x0] * (1 - wx) + a[:, y0][:, :, x1] * wx
    bot = a[:, y1][:, :, x0] * (1 - wx) + a[:, y1][:, :, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def scene_inputs(ds, channels: list[str], f: int, centre: float) -> np.ndarray:
    """(C, H, W) float32, un-normalised, NaN kept. `channels` must list the SAR-grid
    variables first, then the 2 km ones -- the order pick_channels() returns."""
    missing = [c for c in channels if c not in ds]
    if missing:
        raise ValueError(f"scene lacks input variables {missing}")
    sar_vars = [c for c in channels if ds[c].dims == SAR_DIMS]
    amsr_vars = [c for c in channels if ds[c].dims == AMSR_DIMS]
    if sar_vars + amsr_vars != list(channels):
        raise ValueError(f"channels must be SAR-grid variables then 2 km variables: {channels}")
    sar = [block_mean(np.asarray(ds[v].values, dtype=np.float32), f) for v in sar_vars]
    hw = sar[0].shape
    parts = [np.stack(sar)]
    if amsr_vars:
        amsr = np.stack([np.asarray(ds[v].values, dtype=np.float32) for v in amsr_vars])
        parts.append(amsr_to_grid(amsr, hw, f, centre))
    return np.concatenate(parts, axis=0)


def scene_labels(ds, f: int) -> np.ndarray:
    """(3, H, W) uint8 SIC/SOD/FLOE at block centres; FILL = no label."""
    return np.stack([block_centre(np.asarray(ds[l].values).astype(np.uint8), f) for l in LABELS])


def normalise(x: np.ndarray, mean, std) -> np.ndarray:
    """Exactly what the cache stores: (x - mean) / std in float32, NaN -> 0, as float16."""
    mu = np.asarray(mean, dtype=np.float64)[:, None, None].astype(np.float32)
    sd = np.asarray(std, dtype=np.float64)[:, None, None].astype(np.float32)
    return np.nan_to_num((x - mu) / sd, nan=0.0).astype(np.float16)
