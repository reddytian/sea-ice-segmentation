import json

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

import fake_scene  # noqa: E402
from seaice import georef, preprocess  # noqa: E402
from seaice.data import EvalTiles  # noqa: E402
from seaice.model import TASKS, MultiTaskUNet  # noqa: E402
from seaice.predict import load_bundle, main, make_bundle, predict_classes  # noqa: E402


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    p = tmp_path_factory.mktemp("scene") / "20210421T082208_dmi_prep.nc"
    fake_scene.write(p)
    return p


def _tiny_model(in_ch):
    torch.manual_seed(0)
    m = MultiTaskUNet(in_ch=in_ch, tasks=TASKS, base=4)
    with torch.no_grad():                         # non-trivial BatchNorm statistics
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def _bundle(model, channels, mean, std, tile=32):
    return make_bundle(model, in_ch=len(channels), base=4, tasks=TASKS, channels=channels,
                       mean=mean, std=std, factor=5, amsr_spacing=25, amsr_center=12.0,
                       eval_tile=tile, source={"run": "test"})


def test_inference_path_equals_evaluation_path(scene, tmp_path):
    """Write the scene into a cache exactly as build_cache.py does, score it with
    EvalTiles (the path behind every reported metric), and compare with the CLI path."""
    ds = preprocess.open_scene(scene)
    x = preprocess.scene_inputs(ds, fake_scene.CHANNELS, 5, 12.0)
    y = preprocess.scene_labels(ds, 5)
    ds.close()
    mean, std = np.nanmean(x, axis=(1, 2)).tolist(), np.nanstd(x, axis=(1, 2)).tolist()
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    np.save(tmp_path / "x" / "s.npy", preprocess.normalise(x, mean, std))
    np.save(tmp_path / "y" / "s.npy", y)
    (tmp_path / "index.json").write_text(json.dumps({
        "channels": fake_scene.CHANNELS, "labels": ["SIC", "SOD", "FLOE"],
        "scenes": [{"name": "s", "region": "R", "split": "test", "shape": list(y.shape[1:])}]}))

    model = _tiny_model(len(fake_scene.CHANNELS))
    tiles = EvalTiles(tmp_path, "test", tile=32)
    H, W = y.shape[1:]
    ref = {t: np.full((H, W), 255, np.uint8) for t in TASKS}
    with torch.no_grad():
        for k in range(len(tiles)):
            xt, _, _, r0, c0 = tiles[k]
            out = model(xt[None])
            h, w = min(32, H - r0), min(32, W - c0)
            for t in TASKS:
                ref[t][r0:r0 + h, c0:c0 + w] = out[t].argmax(1)[0, :h, :w].numpy()

    got = predict_classes(model, preprocess.normalise(x, mean, std), 32, torch.device("cpu"))
    for t in TASKS:
        assert np.array_equal(got[t], ref[t]), t


def test_georeferencing_accuracy_on_known_swath():
    """Warp row/col index maps; each output pixel must come from the source pixel
    that truly covers its map position (median <= 0.5 px, p99 <= 1 px)."""
    f, H, W = 5, 120, 128                                 # model grid (400 m)
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    tl, ts = np.meshgrid(np.r_[np.arange(0, H * f, 40), H * f - 1],
                         np.r_[np.arange(0, W * f, 40), W * f - 1], indexing="ij")
    x, y = fake_scene.swath_xy(tl + 0.5, ts + 0.5)
    lon, lat = rasterio.warp.transform("EPSG:3413", "EPSG:4326", x.ravel().tolist(), y.ravel().tolist())
    gcps = georef.to_gcps(tl, ts, lat, lon, factor=f)
    # uint8 maps of the source row and column (both < 255; 255 is nodata)
    out, tr = georef.warp({"r": rows.astype(np.uint8), "c": cols.astype(np.uint8)}, gcps)
    ok = out["r"] != 255
    assert ok.sum() > 0.9 * H * W
    rr, cc = np.nonzero(ok)
    xs, ys = rasterio.transform.xy(tr, rr, cc)
    # expected source position: invert swath_xy by Newton iteration, in model pixels
    r = out["r"][ok] + 0.5
    c = out["c"][ok] + 0.5
    for _ in range(6):
        fx, fy = fake_scene.swath_xy(r * f, c * f)
        e = 1e-3
        dxr, dyr = [(a - b) / e for a, b in zip(fake_scene.swath_xy((r + e) * f, c * f), (fx, fy))]
        dxc, dyc = [(a - b) / e for a, b in zip(fake_scene.swath_xy(r * f, (c + e) * f), (fx, fy))]
        det = dxr * dyc - dxc * dyr
        ex, ey = np.asarray(xs) - fx, np.asarray(ys) - fy
        r, c = r + (ex * dyc - ey * dxc) / det, c + (dxr * ey - dyr * ex) / det
    dr, dc = r - (out["r"][ok] + 0.5), c - (out["c"][ok] + 0.5)
    err = np.hypot(dr, dc)
    assert np.median(err) <= 0.5 and np.percentile(err, 99) <= 1.0, (np.median(err), np.percentile(err, 99))
    # No systematic shift: nearest-neighbour errors average out to ~0 over ~15k pixels,
    # so a half-pixel convention slip (0.1 px on this grid) is detectable.
    assert abs(dr.mean()) < 0.03 and abs(dc.mean()) < 0.03, (dr.mean(), dc.mean())


def test_cli_writes_georeferenced_cogs(scene, tmp_path):
    model = _tiny_model(len(fake_scene.CHANNELS))
    bpath = tmp_path / "model.pt"
    torch.save(_bundle(model, fake_scene.CHANNELS, [0.0] * 7, [1.0] * 7), bpath)
    out = tmp_path / "pred"
    assert main([str(scene), "--model", str(bpath), "--out", str(out), "--device", "cpu"]) == 0

    meta = json.loads((out / "20210421T082208_dmi_prep.json").read_text())
    assert set(meta["outputs"]) == {"SIC", "SOD", "FLOE"}
    for t, n in TASKS.items():
        with rasterio.open(out / meta["outputs"][t]) as src:
            assert src.crs.to_epsg() == 3413 and src.res == (400.0, 400.0)
            assert src.transform.c % 400 == 0 and src.transform.f % 400 == 0   # snapped grid
            assert src.nodata == 255 and src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
            a = src.read(1)
            legal = set(range(0, 101, 10)) if t == "SIC" else set(range(n))
            assert set(np.unique(a)) <= legal | {255}
            assert (a != 255).mean() > 0.35       # rotated swath in a north-up box, minus land
            assert src.colormap(1)[0][3] == 255           # colour table present
            # No data where the scene has none (SAR stored as 0 at samples 0-22) or on land;
            # predictions everywhere else in the swath.
            lines = np.arange(40, 280, 40) + 0.5
            for line_set, sample, nodata in ((lines, 8.5, True), (lines + 600, 300.5, False),
                                             (np.arange(980, 1120, 40) + 0.5, 480.5, True)):
                xs, ys = fake_scene.swath_xy(line_set, np.full_like(line_set, sample))
                r, c = rasterio.transform.rowcol(src.transform, xs, ys)
                assert np.all((a[np.asarray(r), np.asarray(c)] == 255) == nodata), (t, sample)
    px = meta["pixels_radar_grid"]
    # 226 x 104 blocks: 5 columns touch the zero strip; land = lines >= 950, samples >= 420
    # by block centre -> 36 x 20 blocks.
    assert px == {"no_sar_data": 226 * 5, "land": 36 * 20, "valid": 226 * 99 - 36 * 20}, px


def test_bundle_loads_without_pickle_and_rejects_other_files(tmp_path):
    model = _tiny_model(7)
    b = _bundle(model, fake_scene.CHANNELS, [0.0] * 7, [1.0] * 7)
    torch.save(b, tmp_path / "ok.pt")
    reloaded, meta = load_bundle(tmp_path / "ok.pt", torch.device("cpu"))
    x = torch.randn(1, 7, 32, 32)
    with torch.no_grad():
        assert all(torch.equal(model(x)[t], reloaded(x)[t]) for t in TASKS)
    torch.save({"model": model.state_dict()}, tmp_path / "raw.pt")
    with pytest.raises(ValueError, match="bundle"):
        load_bundle(tmp_path / "raw.pt", torch.device("cpu"))
    with pytest.raises(ValueError, match="channels"):
        _bundle(model, fake_scene.CHANNELS[:3], [0.0] * 7, [1.0] * 7)


def test_grid_node_positions_follow_the_measured_layout():
    """Line nodes every 250 lines plus the last line (node count varies with scene
    length); sample nodes even. Cases taken from real scene shapes."""
    from seaice.georef import node_positions
    for n_nodes, n_lines in ((21, 4996), (23, 5353), (7, 1391), (21, 5002), (6, 1200)):
        lines, samples = node_positions(n_nodes, 21, n_lines, 5242)
        assert lines[1] == 250 and lines[-1] == n_lines - 1 and len(lines) == n_nodes
        assert samples[0] == 0 and samples[-1] == 5241 and np.allclose(np.diff(samples), 5241 / 20)
    with pytest.raises(ValueError):
        node_positions(21, 21, 5353, 5242)          # too few rows for this many lines
    with pytest.raises(ValueError):
        node_positions(23, 21, 4996, 5242)          # too many
