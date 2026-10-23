#!/usr/bin/env python3
"""Full-corpus label statistics for the AI4Arctic subset.

Why scan all 123 scenes rather than sample a few: the three-scene inspection
reported SOD class 1 at 0.020% and class 3 at 0.017%. Those numbers cannot
distinguish "globally rare" from "absent from these three scenes", and the
distinction decides the loss function. The same inspection also showed that the
distinct value SET differs between scenes, so a per-scene histogram is the only
honest input to a class-weight calculation.

One pass over 13 GB, cached to corpus_stats.npz. Nothing downstream re-reads
the raw scenes to ask a question about class frequency.

Opened with mask_and_scale=False: xarray's default would promote the uint8
class labels to float32 and turn fill into NaN, which is the wrong
representation for CrossEntropyLoss targets.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    sys.exit("needs xarray: pip install xarray netcdf4")

LABELS = ("SIC", "SOD", "FLOE")
NBINS = 256          # uint8 labels, so a fixed-width histogram covers every value


def parse_name(name: str) -> tuple[str, str]:
    """(acquisition datetime, ice service) from the ready-to-train filename."""
    m = re.match(r"(\d{8}T\d{6})_(\w+?)_prep", name)
    return (m.group(1), m.group(2)) if m else ("", "")


def load_regions(manifest: Path) -> dict[str, str]:
    reg = {}
    if not manifest.exists():
        return reg
    with open(manifest) as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 4 and p[0] != "file":
                reg[p[0]] = p[3]
    return reg


def scan_one(path: Path) -> dict:
    """Per-scene histograms plus the geometry facts the Dataset class needs."""
    ds = xr.open_dataset(path, mask_and_scale=False)
    out = {}
    for lab in LABELS:
        if lab in ds:
            a = np.asarray(ds[lab].values)
            if a.dtype != np.uint8:
                a = a.astype(np.uint8)
            out[lab] = np.bincount(a.ravel(), minlength=NBINS)[:NBINS]
        else:
            out[lab] = np.zeros(NBINS, dtype=np.int64)

    sz = dict(ds.sizes)
    out["sar_shape"] = (sz.get("sar_lines", 0), sz.get("sar_samples", 0))
    out["amsr_shape"] = (sz.get("2km_grid_lines", 0), sz.get("2km_grid_samples", 0))

    # The SAR:AMSR2 ratio was 25.00-25.11 on the three sampled scenes, i.e. not
    # an integer. If it varies across the corpus, the Dataset class cannot use a
    # fixed upsampling factor and has to interpolate per scene.
    out["ratio"] = (
        out["sar_shape"][0] / out["amsr_shape"][0] if out["amsr_shape"][0] else 0.0,
        out["sar_shape"][1] / out["amsr_shape"][1] if out["amsr_shape"][1] else 0.0,
    )

    # NaN in the SAR channels would have to be handled separately from the
    # label fill value; the sample said there is none, worth confirming at scale.
    nan_sar = 0
    for v in ("nersc_sar_primary", "nersc_sar_secondary"):
        if v in ds:
            nan_sar += int(np.isnan(np.asarray(ds[v].values)).sum())
    out["nan_sar"] = nan_sar

    out["vars"] = sorted(map(str, ds.variables))
    ds.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="~/dl-local/ai4arctic/scenes")
    ap.add_argument("--manifest", default="~/dl-local/ai4arctic/manifest.csv")
    ap.add_argument("--out", default="~/dl-local/ai4arctic/corpus_stats.npz")
    args = ap.parse_args()

    d = Path(os.path.expanduser(args.dir))
    files = sorted(d.glob("*.nc"))
    if not files:
        sys.exit(f"no .nc files in {d}")
    regions = load_regions(Path(os.path.expanduser(args.manifest)))

    print(f"scanning {len(files)} scenes in {d}")
    names, acqs, srcs, regs = [], [], [], []
    hist = {lab: [] for lab in LABELS}
    sar_shapes, amsr_shapes, ratios, nan_sars = [], [], [], []
    var_sets = []

    t0 = time.time()
    for i, f in enumerate(files, 1):
        r = scan_one(f)
        acq, src = parse_name(f.name)
        names.append(f.name)
        acqs.append(acq)
        srcs.append(src)
        regs.append(regions.get(f.name, "?"))
        for lab in LABELS:
            hist[lab].append(r[lab])
        sar_shapes.append(r["sar_shape"])
        amsr_shapes.append(r["amsr_shape"])
        ratios.append(r["ratio"])
        nan_sars.append(r["nan_sar"])
        var_sets.append(set(r["vars"]))
        el = time.time() - t0
        print(f"\r  {i:3d}/{len(files)}  {el:6.1f}s  "
              f"eta {el/i*(len(files)-i):6.1f}s", end="", flush=True)
    print(f"\n  done in {time.time()-t0:.1f}s\n")

    H = {lab: np.array(hist[lab], dtype=np.int64) for lab in LABELS}

    # ---- which values actually occur, corpus-wide ----
    print("=" * 74)
    print("label values, whole corpus")
    print("=" * 74)
    for lab in LABELS:
        tot = H[lab].sum(axis=0)
        present = np.nonzero(tot)[0]
        valid = tot[:255].sum()           # 255 is fill, excluded from the denominator
        print(f"\n{lab}: values {present.tolist()}")
        print(f"  {'val':>4s} {'pixels':>16s} {'% of valid':>11s} {'scenes w/ it':>13s}")
        for v in present:
            n_sc = int((H[lab][:, v] > 0).sum())
            pct = tot[v] / valid * 100 if v != 255 and valid else float("nan")
            tag = "  <- FILL" if v == 255 else ""
            pcts = "       --  " if v == 255 else f"{pct:10.4f}%"
            print(f"  {v:4d} {tot[v]:16,d} {pcts} {n_sc:10d}/{len(files)}{tag}")

    # ---- how rare is rare, and in how many scenes ----
    print("\n" + "=" * 74)
    print("rarity: a class present in few scenes cannot be validated, however")
    print("many pixels it has -- all of them may sit inside one split")
    print("=" * 74)
    for lab in ("SIC", "SOD"):
        tot = H[lab].sum(axis=0)
        valid = tot[:255].sum()
        present = [v for v in np.nonzero(tot)[0] if v != 255]
        print(f"\n{lab}")
        for v in present:
            col = H[lab][:, v]
            n_sc = int((col > 0).sum())
            # scenes holding 90% of this class's pixels
            order = np.argsort(col)[::-1]
            csum = np.cumsum(col[order])
            n90 = int(np.searchsorted(csum, 0.9 * col.sum()) + 1)
            print(f"  val {v:3d}  {col.sum()/valid*100:8.4f}% of valid  "
                  f"in {n_sc:3d} scenes  90% of it in {n90:3d} scene(s)")

    # ---- valid fraction per scene ----
    print("\n" + "=" * 74)
    print("fill fraction per scene (SIC)")
    print("=" * 74)
    sic = H["SIC"]
    frac_fill = sic[:, 255] / sic.sum(axis=1)
    for p in (0, 25, 50, 75, 100):
        print(f"  {p:3d}th pct  {np.percentile(frac_fill, p)*100:6.2f}%")
    worst = np.argsort(frac_fill)[::-1][:5]
    print("  most-filled scenes:")
    for i in worst:
        print(f"    {names[i]:32s} {frac_fill[i]*100:6.2f}% fill  {regs[i]}")

    # ---- do the three labels agree on where fill is ----
    print("\nfill agreement between labels (per-scene fill pixel counts):")
    same = sum(1 for i in range(len(files))
               if H["SIC"][i, 255] == H["SOD"][i, 255] == H["FLOE"][i, 255])
    print(f"  SIC/SOD/FLOE fill counts identical in {same}/{len(files)} scenes")
    if same != len(files):
        print("  -> the three tasks have DIFFERENT valid masks; each head needs its own")

    # ---- geometry ----
    print("\n" + "=" * 74)
    print("geometry")
    print("=" * 74)
    ss = np.array(sar_shapes)
    az = np.array(amsr_shapes)
    rr = np.array(ratios)
    print(f"  sar_lines   {ss[:,0].min()}-{ss[:,0].max()}   "
          f"sar_samples {ss[:,1].min()}-{ss[:,1].max()}")
    print(f"  2km lines   {az[:,0].min()}-{az[:,0].max()}   "
          f"2km samples {az[:,1].min()}-{az[:,1].max()}")
    print(f"  SAR:AMSR ratio  lines {rr[:,0].min():.3f}-{rr[:,0].max():.3f}  "
          f"samples {rr[:,1].min():.3f}-{rr[:,1].max():.3f}")
    print(f"  -> {'non-integer and variable: interpolate per scene' if rr.std() > 1e-6 else 'constant'}")
    print(f"  NaN in nersc_sar_* : {sum(nan_sars):,} px across all scenes")
    smallest = ss.min(axis=0)
    print(f"  smallest scene {tuple(smallest)} -> crop size must be <= {smallest.min()}")

    common = set.intersection(*var_sets)
    allv = set.union(*var_sets)
    print(f"\n  variables: {len(common)} in every scene, {len(allv)} overall")
    if allv - common:
        print("  NOT in every scene:")
        for v in sorted(allv - common):
            n = sum(1 for s in var_sets if v in s)
            print(f"    {v:28s} in {n}/{len(files)}")

    out = Path(os.path.expanduser(args.out))
    np.savez_compressed(
        out,
        names=np.array(names), acq=np.array(acqs), source=np.array(srcs),
        region=np.array(regs), sar_shape=ss, amsr_shape=az, ratio=rr,
        nan_sar=np.array(nan_sars),
        **{f"hist_{lab}": H[lab] for lab in LABELS},
    )
    print(f"\ncached -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
