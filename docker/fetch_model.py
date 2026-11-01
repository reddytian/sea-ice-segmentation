#!/usr/bin/env python3
"""Download the published model bundle and verify its SHA-256 before keeping it.

    python docker/fetch_model.py OUT.pt [--spec docker/model.json]

docker/model.json (written by scripts/export_model.py --release TAG) records the
release URL and the checksum, so the image can only ever contain the exact bundle
that was exported and checked against the evaluation pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--spec", default=str(Path(__file__).with_name("model.json")))
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(spec["url"], timeout=120) as r, open(tmp, "wb") as fh:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            fh.write(chunk)
    if h.hexdigest() != spec["sha256"]:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {spec['url']}: got {h.hexdigest()}, expected {spec['sha256']}")
    os.replace(tmp, out)
    print(f"{out}  sha256 {spec['sha256'][:12]}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
