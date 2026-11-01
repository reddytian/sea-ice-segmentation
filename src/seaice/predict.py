"""seaice-predict: AI4Arctic scene -> georeferenced ice maps (Cloud-Optimised GeoTIFF).

    seaice-predict SCENE.nc [SCENE.nc ...] --model seaice_unet_r2.pt --out predictions/

Per scene it writes single-band uint8 COGs in EPSG:3413 at 400 m (255 = no data,
colour table embedded, so they open coloured in QGIS):
    <scene>_SIC.tif    sea ice concentration, percent (0, 10, ..., 100)
    <scene>_SOD.tif    stage of development, class code (legend in the JSON)
    <scene>_FLOE.tif   floe size, class code
    <scene>.json       provenance (model, checksum, inputs), legend, pixel counts, timing

Input is the AI4Arctic ready-to-train NetCDF format: calibrated Sentinel-1 EW HH/HV
on the swath grid plus AMSR2 brightness temperatures on a 2 km grid. Producing that
from raw Sentinel-1 GRD and AMSR2 L1R is upstream preprocessing, not done here.

The inference path is the evaluation path: same preprocessing module as the training
cache (including float16 rounding), same non-overlapping zero-padded tiles as
EvalTiles. scripts/check_parity.py verifies both on real scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import georef, preprocess
from .model import MultiTaskUNet

FORMAT = "seaice-model/1"
FILL = 255
LAND_VALUE = preprocess.LAND_VALUE
CLASSES = {
    "SIC": [f"{10 * k}%" for k in range(11)],
    "SOD": ["open water", "new ice", "young ice", "thin first-year ice",
            "thick first-year ice", "old ice"],
    "FLOE": ["open water", "cake ice", "small floe", "medium floe", "big floe",
             "vast floe", "iceberg"],
}
# Same palettes as the README figures (validated sequential/ordinal ramps).
_SIC = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
_SOD = ["#cde2fb", "#ff8e64", "#ea6b3a", "#cd501a", "#aa3b01", "#842c00"]
_FLOE = ["#cde2fb", "#e8e3f6", "#cbbfee", "#ad9ce4", "#8f79d9", "#7156cd", "#3d2a85"]


def _rgba(h: str) -> tuple[int, int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), 255


# ---------------------------------------------------------------- model bundle
def make_bundle(model: MultiTaskUNet, *, in_ch: int, base: int, tasks: dict[str, int],
                channels: list[str], mean: list[float], std: list[float], factor: int,
                amsr_spacing: int, amsr_center: float, eval_tile: int, source: dict) -> dict:
    """Everything inference needs, nothing it does not: weights, architecture,
    input channels in order, train-only normalisation, preprocessing constants."""
    if not (len(channels) == len(mean) == len(std) == in_ch):
        raise ValueError(f"{len(channels)} channels, {len(mean)} means, {len(std)} stds, in_ch {in_ch}")
    if amsr_spacing != preprocess.AMSR_SPACING:
        raise ValueError(f"model expects AMSR spacing {amsr_spacing}, code has {preprocess.AMSR_SPACING}")
    return {
        "format": FORMAT,
        "arch": {"name": "MultiTaskUNet", "in_ch": in_ch, "base": base, "tasks": dict(tasks)},
        "state_dict": {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()},
        "channels": list(channels),
        "norm": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
        "preprocess": {"factor": int(factor), "amsr_spacing": int(amsr_spacing),
                       "amsr_center": float(amsr_center)},
        "eval_tile": int(eval_tile),
        "classes": CLASSES,
        "source": dict(source),
    }


def load_bundle(path: str | os.PathLike, device: torch.device):
    """A bundle holds only tensors and plain Python values, so it loads with
    weights_only=True: opening a downloaded model file cannot execute code."""
    b = torch.load(path, map_location="cpu", weights_only=True)
    if b.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} bundle (format={b.get('format')!r})")
    a = b["arch"]
    model = MultiTaskUNet(in_ch=a["in_ch"], tasks=a["tasks"], base=a["base"])
    model.load_state_dict(b["state_dict"])
    return model.to(device).eval(), b


def sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_classes(model, x16: np.ndarray, tile: int, device, batch: int = 4) -> dict[str, np.ndarray]:
    """Argmax class per task for a (C, H, W) float16 normalised scene.

    Mirrors EvalTiles + engine.evaluate: non-overlapping tile x tile windows from
    (0, 0), zero-padded past the edge, float16 inputs widened to float32.
    """
    C, H, W = x16.shape
    out = {t: np.full((H, W), FILL, np.uint8) for t in model.heads}
    windows = [(r0, c0) for r0 in range(0, H, tile) for c0 in range(0, W, tile)]
    for k in range(0, len(windows), batch):
        chunk = windows[k:k + batch]
        xb = np.zeros((len(chunk), C, tile, tile), np.float32)
        for j, (r0, c0) in enumerate(chunk):
            h, w = min(tile, H - r0), min(tile, W - c0)
            xb[j, :, :h, :w] = x16[:, r0:r0 + h, c0:c0 + w]
        logits = model(torch.from_numpy(xb).to(device))
        for t in out:
            pred = logits[t].argmax(1).to("cpu").numpy().astype(np.uint8)
            for j, (r0, c0) in enumerate(chunk):
                h, w = min(tile, H - r0), min(tile, W - c0)
                out[t][r0:r0 + h, c0:c0 + w] = pred[j, :h, :w]
    return out


def valid_mask(ds, f: int) -> tuple[np.ndarray, dict[str, int]]:
    """Model-grid pixels to report: every full-resolution pixel of the block has SAR
    data (outside the swath both channels are stored as exactly 0), and the block
    centre is not land (distance_map). The model never saw a label on land."""
    hh = np.asarray(ds["nersc_sar_primary"].values)
    hv = np.asarray(ds["nersc_sar_secondary"].values)
    has_sar = np.isfinite(hh) & np.isfinite(hv) & ((hh != 0) | (hv != 0))
    in_swath = preprocess.block_mean(has_sar.astype(np.float32), f) == 1.0
    land = np.zeros_like(in_swath)
    if "distance_map" in ds:
        dm = np.asarray(ds["distance_map"].values)
        land = preprocess.block_centre(np.abs(dm - LAND_VALUE) < 1e-4, f)
    ok = in_swath & ~land
    # The two overlap: AI4Arctic files also store SAR as 0 over land.
    return ok, {"no_sar_data": int((~in_swath).sum()), "land": int(land.sum()), "valid": int(ok.sum())}


def tie_points(ds):
    """(lines, samples, lats, lons): the scene's lat/lon grid with each node's
    full-resolution pixel position (see georef.GRID_CONVENTION)."""
    if "sar_grid2d_latitude" not in ds or "sar_grid2d_longitude" not in ds:
        raise ValueError("scene has no sar_grid2d_latitude/longitude")
    lat = georef.grid_to_line_sample(ds["sar_grid2d_latitude"].values, *georef.GRID_CONVENTION)
    lon = georef.grid_to_line_sample(ds["sar_grid2d_longitude"].values, *georef.GRID_CONVENTION)
    kl, ks = georef.node_positions(*lat.shape, ds.sizes["sar_lines"], ds.sizes["sar_samples"])
    lines, samples = np.meshgrid(kl, ks, indexing="ij")
    return lines, samples, lat, lon


def run_scene(path: Path, model, bundle: dict, out_dir: Path, device, crs: str, res: float,
              model_sha: str) -> dict:
    t0 = time.time()
    pp = bundle["preprocess"]
    channels = bundle["channels"]
    ds = preprocess.open_scene(path)
    try:
        x_raw = preprocess.scene_inputs(ds, channels, pp["factor"], pp["amsr_center"])
        tp = tie_points(ds)
        valid, counts = valid_mask(ds, pp["factor"])
    finally:
        ds.close()
    x16 = preprocess.normalise(x_raw, bundle["norm"]["mean"], bundle["norm"]["std"])
    t_prep = time.time() - t0

    preds = predict_classes(model, x16, bundle["eval_tile"], device)
    t_model = time.time() - t0 - t_prep
    preds["SIC"] = preds["SIC"] * np.uint8(10)          # class k -> k*10 percent
    for t in preds:
        preds[t][~valid] = FILL

    gcps = georef.to_gcps(*tp, factor=pp["factor"], crs=crs)
    warped, transform = georef.warp(preds, gcps, crs=crs, res=res, nodata=FILL)
    t_warp = time.time() - t0 - t_prep - t_model

    stem = path.name[:-3] if path.name.endswith(".nc") else path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    palettes = {"SIC": {10 * k: _rgba(c) for k, c in enumerate(_SIC)},
                "SOD": {k: _rgba(c) for k, c in enumerate(_SOD)},
                "FLOE": {k: _rgba(c) for k, c in enumerate(_FLOE)}}
    for t, a in warped.items():
        f = out_dir / f"{stem}_{t}.tif"
        codes = list(palettes[t])
        georef.write_cog(f, a, transform, crs, FILL, palettes[t], description=t,
                         tags={"classes": json.dumps(dict(zip(codes, CLASSES[t])))})
        files[t] = f.name
    meta = {
        "scene": path.name,
        "outputs": files,
        "crs": crs, "resolution_m": res, "nodata": FILL,
        "legend": {"SIC": "percent", "SOD": dict(enumerate(CLASSES["SOD"])),
                   "FLOE": dict(enumerate(CLASSES["FLOE"]))},
        "pixels_radar_grid": counts,
        "class_fractions": {t: {str(int(v)): round(float(np.mean(preds[t][valid] == v)), 4)
                                for v in np.unique(preds[t][valid])} for t in preds},
        "model": {"sha256": model_sha, **bundle["source"]},
        "inputs": {"channels": channels, **pp},
        "seconds": {"preprocess": round(t_prep, 2), "model": round(t_model, 2),
                    "warp": round(t_warp, 2), "total": round(time.time() - t0, 2)},
        "device": str(device),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seaice-predict", description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", nargs="+", help="AI4Arctic ready-to-train NetCDF file(s)")
    ap.add_argument("--model", default=os.environ.get("SEAICE_MODEL"),
                    help="model bundle (.pt); default $SEAICE_MODEL")
    ap.add_argument("--out", default="predictions", help="output directory")
    ap.add_argument("--crs", default="EPSG:3413", help="output CRS (default NSIDC polar stereographic north)")
    ap.add_argument("--res", type=float, default=400.0, help="output pixel size in CRS units")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    args = ap.parse_args(argv)
    if not args.model:
        ap.error("--model is required (or set SEAICE_MODEL)")

    device = pick_device(args.device)
    model, bundle = load_bundle(args.model, device)
    model_sha = sha256(args.model)
    print(f"model {Path(args.model).name} sha256 {model_sha[:12]} "
          f"({bundle['source'].get('run', '?')}) on {device}", file=sys.stderr)
    failed = 0
    for s in args.scenes:
        p = Path(s)
        try:
            m = run_scene(p, model, bundle, Path(args.out), device, args.crs, args.res, model_sha)
            print(f"{p.name}: {', '.join(m['outputs'].values())}  ({m['seconds']['total']} s)")
        except Exception as e:                       # one bad scene must not stop a batch
            failed += 1
            print(f"{p.name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
