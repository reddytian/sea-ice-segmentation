import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FILL = 255
CHANNELS = ["nersc_sar_primary", "nersc_sar_secondary", "sar_incidenceangle", "btemp_6_9h"]


@pytest.fixture
def tiny_cache(tmp_path):
    """A cache in build_cache.py's layout: 5 scenes, 3 regions, 4 channels.

    Channel 0 of x equals the SIC label wherever SIC is labelled, so any crop,
    flip or rotation that misaligns inputs and labels is detectable.
    """
    rng = np.random.default_rng(0)
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    plan = [("20190101T000000_dmi_prep", "RA", "train", (40, 37)),
            ("20190201T000000_dmi_prep", "RA", "train", (20, 20)),     # smaller than a 32 crop
            ("20190301T000000_cis_prep", "RB", "train", (45, 50)),
            ("20190401T000000_dmi_prep", "RC", "val", (33, 41)),
            ("20190501T000000_cis_prep", "RD", "test", (30, 30))]
    scenes = []
    for name, region, split, (H, W) in plan:
        y = np.stack([rng.integers(0, 11, (H, W)), rng.integers(0, 6, (H, W)),
                      rng.integers(0, 7, (H, W))]).astype(np.uint8)
        y[:, :3] = FILL                                                  # an unlabelled band
        x = rng.normal(size=(len(CHANNELS), H, W)).astype(np.float16)
        x[0] = np.where(y[0] == FILL, 0, y[0]).astype(np.float16)
        np.save(tmp_path / "x" / f"{name}.npy", x)
        np.save(tmp_path / "y" / f"{name}.npy", y)
        scenes.append({"name": name, "region": region, "split": split, "shape": [H, W],
                       "nan_inputs": 0,
                       "valid_frac": {l: float((y[k] != FILL).mean())
                                      for k, l in enumerate(("SIC", "SOD", "FLOE"))}})
    (tmp_path / "index.json").write_text(json.dumps(
        {"channels": CHANNELS, "labels": ["SIC", "SOD", "FLOE"], "fill": FILL, "scenes": scenes}))
    return tmp_path
