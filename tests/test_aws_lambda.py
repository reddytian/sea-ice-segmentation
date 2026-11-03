"""The Lambda handler, with S3 replaced by a local fake (no AWS, no boto3 needed)."""
import json
import shutil

import pytest
import torch

pytest.importorskip("rasterio")

import fake_scene  # noqa: E402
from seaice import aws_lambda  # noqa: E402
from seaice.model import TASKS, MultiTaskUNet  # noqa: E402
from seaice.predict import make_bundle  # noqa: E402


class FakeS3:
    def __init__(self, root):
        self.root = root

    def download_file(self, bucket, key, path):
        shutil.copy(self.root / bucket / key, path)

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        dst = self.root / bucket / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dst)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        dst = self.root / Bucket / Key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Body)


def _event(*keys, bucket="b"):
    return {"Records": [{"eventSource": "aws:s3", "s3": {"bucket": {"name": bucket},
                                                          "object": {"key": k}}} for k in keys]}


@pytest.fixture
def lambda_env(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = MultiTaskUNet(in_ch=7, tasks=TASKS, base=4).eval()
    bundle = make_bundle(model, in_ch=7, base=4, tasks=TASKS, channels=fake_scene.CHANNELS,
                         mean=[0.0] * 7, std=[1.0] * 7, factor=5, amsr_spacing=25, amsr_center=12.0,
                         eval_tile=32, source={"run": "test"})
    torch.save(bundle, tmp_path / "model.pt")
    (tmp_path / "s3" / "b" / "incoming").mkdir(parents=True)
    monkeypatch.setenv("SEAICE_MODEL", str(tmp_path / "model.pt"))
    monkeypatch.setattr(aws_lambda, "WORK", tmp_path / "work")
    monkeypatch.setattr(aws_lambda, "_state", {"s3": FakeS3(tmp_path / "s3")})
    return tmp_path / "s3" / "b"


def test_scene_in_s3_becomes_cogs_in_s3(lambda_env):
    fake_scene.write(lambda_env / "incoming" / "20210421T082208_dmi_prep.nc")
    out = aws_lambda.handler(_event("incoming/20210421T082208_dmi_prep.nc"))
    assert [r["status"] for r in out["results"]] == ["ok"]
    got = sorted(p.name for p in (lambda_env / "predictions" / "20210421T082208_dmi_prep").iterdir())
    assert got == ["20210421T082208_dmi_prep.json", "20210421T082208_dmi_prep_FLOE.tif",
                   "20210421T082208_dmi_prep_SIC.tif", "20210421T082208_dmi_prep_SOD.tif"]
    assert not any((aws_lambda.WORK).rglob("*.nc"))          # /tmp cleaned up


def test_bad_scene_is_reported_and_does_not_stop_the_batch(lambda_env):
    fake_scene.write(lambda_env / "incoming" / "good.nc")
    (lambda_env / "incoming" / "broken.nc").write_bytes(b"not a netcdf file")
    out = aws_lambda.handler(_event("incoming/broken.nc", "incoming/good.nc", "incoming/readme.txt"))
    assert [r["status"] for r in out["results"]] == ["error", "ok"]      # .txt ignored
    err = json.loads((lambda_env / "predictions" / "broken" / "ERROR.json").read_text())
    assert err["scene"] == "incoming/broken.nc" and err["error"]


def test_module_import_stays_within_lambda_init_budget():
    """Lambda caps the init phase (module import) at 10 s; importing torch there timed
    out on the first deployment. The handler module must not pull in torch or rasterio."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    src = str(Path(aws_lambda.__file__).resolve().parents[1])
    code = "import sys, seaice.aws_lambda; print(sorted({'torch', 'rasterio'} & set(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": src})
    assert out.stdout.strip() == "[]", out.stdout


def test_cold_start_is_reported_once(lambda_env):
    fake_scene.write(lambda_env / "incoming" / "a.nc")
    first = aws_lambda.handler(_event("incoming/a.nc"))
    second = aws_lambda.handler(_event("incoming/a.nc"))
    assert first["cold_start"] is True and second["cold_start"] is False
    assert first["import_s"] is not None and first["model_load_s"] is not None


def test_url_encoded_keys_are_decoded(lambda_env):
    fake_scene.write(lambda_env / "incoming" / "scene with space.nc")
    out = aws_lambda.handler(_event("incoming/scene+with+space.nc"))
    assert out["results"][0]["status"] == "ok"
