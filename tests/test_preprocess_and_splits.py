import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from build_cache import AMSR_SPACING, amsr_to_grid, block_centre, block_mean

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_block_mean_and_centre():
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert block_mean(a, 5)[0, 0] == a[:5, :5].mean()
    assert block_centre(a, 5)[1, 1] == a[7, 7]
    assert block_mean(np.ones((12, 13)), 5).shape == (2, 2)


def test_amsr_resampled_at_fixed_spacing_not_stretched():
    """Cell k is centred at SAR pixel 25k + 12. Encode each cell's centre coordinate
    as its value; the output must equal each output pixel's own centre, clamped
    to the AMSR grid -- including scenes with 0-25 spare SAR pixels at the edge."""
    f, centre = 5, 12.0
    for n_sar in (250, 262, 275):
        h = n_sar // AMSR_SPACING
        rows = (np.arange(h) * AMSR_SPACING + centre).astype(np.float32)
        grid = np.repeat(rows[None, :, None], 3, axis=2)              # (1, h, 3)
        out = amsr_to_grid(grid, (n_sar // f, 1), f, centre)[0, :, 0]
        expected = np.clip(np.arange(n_sar // f) * f + (f - 1) / 2,
                           centre, AMSR_SPACING * (h - 1) + centre)
        assert np.allclose(out, expected, atol=1e-3)


def _run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], check=True,
                   capture_output=True)


def _bigger_cache(tmp_path):
    """Index-only cache: the split scripts read nothing but index.json."""
    scenes, k = [], 0
    for region, split, n in [("RA", "train", 6), ("RB", "train", 6),
                             ("RC", "val", 4), ("RD", "val", 4), ("RE", "test", 3)]:
        for _ in range(n):
            k += 1
            scenes.append({"name": f"2019{1 + k % 12:02d}{1 + k % 27:02d}T{k:06d}_dmi_prep",
                           "region": region, "split": split})
    (tmp_path / "index.json").write_text(json.dumps({"scenes": scenes}))
    return tmp_path, scenes


def test_random_split_keeps_test_and_sizes(tmp_path):
    cache, scenes = _bigger_cache(tmp_path)
    out = tmp_path / "rand.json"
    _run("make_random_split.py", "--cache", cache, "--out", out)
    m = json.loads(out.read_text())["scenes"]
    assert all(m[s["name"]] == "test" for s in scenes if s["split"] == "test")
    for sp in ("train", "val"):
        assert sum(v == sp for v in m.values()) == sum(s["split"] == sp for s in scenes)


def test_leak_split_differs_only_in_swapped_scenes(tmp_path):
    cache, scenes = _bigger_cache(tmp_path)
    _run("make_leak_split.py", "--cache", cache, "--out-dir", tmp_path)
    a = json.loads((tmp_path / "split_leakA.json").read_text())
    b = json.loads((tmp_path / "split_leakB.json").read_text())
    d = a["design"]
    ta = {k for k, v in a["scenes"].items() if v == "train"}
    tb = {k for k, v in b["scenes"].items() if v == "train"}
    E, S, D = set(d["E_eval"]), set(d["S_same_region_other_dates"]), set(d["D_dropped_from_B"])
    assert len(ta) == len(tb)
    assert tb == (ta - D) | S
    assert not E & (ta | tb)
    assert all(a["scenes"][n] == b["scenes"][n] == "val" for n in E)
    assert all(a["scenes"][s["name"]] == "test" for s in scenes if s["split"] == "test")
