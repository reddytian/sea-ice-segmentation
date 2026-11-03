"""AWS Lambda entry point: an AI4Arctic scene lands in S3, maps come back to S3.

    s3://BUCKET/incoming/<scene>.nc  --(S3 event)-->  this handler
      -> s3://BUCKET/predictions/<scene>/<scene>_{SIC,SOD,FLOE}.tif + <scene>.json

The handler runs the same `seaice.predict.run_scene` as the CLI and the Docker image,
with the model bundle baked into the image (SEAICE_MODEL). The model is loaded once per
container (cold start) and reused while Lambda keeps the container warm.

Only /tmp is writable in Lambda; each scene is downloaded there, processed and removed,
so a warm container does not fill its ephemeral storage. The S3 trigger is limited to
`incoming/*.nc` and outputs go under `predictions/`, so the function cannot trigger
itself. A scene that fails writes `predictions/<scene>/ERROR.json` and the next scene in
the same event still runs.

torch, rasterio and the model are imported on the first invocation, not when Lambda
loads this module. Lambda's init phase is capped at 10 s; on the first deployment,
importing torch at module level hit that cap ("INIT_REPORT ... Status: timeout"), and
Lambda then repeated the whole init inside the invocation. Deferring the imports pays
that cost once.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path
from urllib.parse import unquote_plus

OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "predictions/")
CRS = os.environ.get("SEAICE_CRS", "EPSG:3413")
RES = float(os.environ.get("SEAICE_RES", "400"))
WORK = Path(os.environ.get("SEAICE_WORKDIR", "/tmp/seaice"))

_state: dict = {}


def _model():
    """Import the inference stack and load the bundle, once per container."""
    if "model" not in _state:
        t0 = time.time()
        import torch

        from .predict import load_bundle, run_scene, sha256
        t1 = time.time()
        path = os.environ["SEAICE_MODEL"]
        model, bundle = load_bundle(path, torch.device("cpu"))
        _state.update(model=model, bundle=bundle, sha=sha256(path), run_scene=run_scene,
                      device=torch.device("cpu"), import_s=round(t1 - t0, 2),
                      load_s=round(time.time() - t1, 2))
    return _state["model"], _state["bundle"], _state["sha"]


def _s3():
    if "s3" not in _state:
        import boto3
        _state["s3"] = boto3.client("s3")
    return _state["s3"]


def _scene_records(event: dict):
    for rec in event.get("Records", []):
        if rec.get("eventSource") != "aws:s3":
            continue
        key = unquote_plus(rec["s3"]["object"]["key"])
        if key.endswith(".nc"):
            yield rec["s3"]["bucket"]["name"], key


def handler(event, context=None):
    cold = "model" not in _state
    model, bundle, sha = _model()
    s3 = _s3()
    results = []
    for bucket, key in _scene_records(event):
        name = Path(key).name
        stem = name[:-3]
        dest = f"{OUTPUT_PREFIX}{stem}/"
        job = WORK / stem
        shutil.rmtree(job, ignore_errors=True)
        (job / "out").mkdir(parents=True)
        t0 = time.time()
        try:
            local = job / name
            s3.download_file(bucket, key, str(local))
            meta = _state["run_scene"](local, model, bundle, job / "out", _state["device"], CRS, RES, sha)
            local.unlink()                                  # free /tmp before uploading
            for f in sorted((job / "out").iterdir()):
                ctype = "image/tiff" if f.suffix == ".tif" else "application/json"
                s3.upload_file(str(f), bucket, dest + f.name, ExtraArgs={"ContentType": ctype})
            results.append({"scene": key, "status": "ok", "outputs": f"s3://{bucket}/{dest}",
                            "seconds": round(time.time() - t0, 2), "model_s": meta["seconds"]})
        except Exception as e:                              # one bad scene must not stop the rest
            err = {"scene": key, "status": "error", "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-4000:]}
            s3.put_object(Bucket=bucket, Key=dest + "ERROR.json", Body=json.dumps(err, indent=1).encode(),
                          ContentType="application/json")
            results.append({k: v for k, v in err.items() if k != "traceback"})
        finally:
            shutil.rmtree(job, ignore_errors=True)
    out = {"model_sha256": sha, "cold_start": cold, "import_s": _state.get("import_s"),
           "model_load_s": _state.get("load_s"), "results": results}
    print(json.dumps(out))                                  # -> CloudWatch Logs
    return out
