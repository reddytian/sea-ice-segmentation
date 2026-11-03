"""Run the Lambda handler inside the `lambda` image under Lambda's constraints, without AWS.

    docker build --platform linux/arm64 --target lambda -t seaice-lambda:local .
    docker run --rm --read-only --tmpfs /tmp:rw,exec,size=4g --user 993:990 \\
      -v "$PWD/deploy/aws:/check:ro" -v ~/dl-local/ai4arctic/scenes:/scenes:ro \\
      -v "$PWD/predictions:/out" \\
      --entrypoint python seaice-lambda:local /check/local_check.py 20210421T082208_dmi_prep.nc

As in Lambda: a read-only root file system, only /tmp writable, an arbitrary non-root
user. S3 is replaced by folders: /scenes plays s3://BUCKET/incoming/, and
/out/lambda_local plays s3://BUCKET/predictions/. Everything else -- the handler, the
model baked into the image, /tmp handling -- is what runs in AWS.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

SCENES = Path(os.environ.get("CHECK_SCENES", "/scenes"))
OUT = Path(os.environ.get("CHECK_OUT", "/out/lambda_local"))


class FolderS3:
    """The three S3 calls the handler makes, against local folders."""

    def download_file(self, bucket, key, path):
        shutil.copyfile(SCENES / Path(key).name, path)

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        dst = OUT / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        dst = OUT / Key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Body)


def main() -> int:
    t0 = time.time()
    from seaice import aws_lambda                        # Lambda's init phase: must stay light
    t_init = time.time() - t0
    aws_lambda._state["s3"] = FolderS3()
    event = {"Records": [{"eventSource": "aws:s3",
                          "s3": {"bucket": {"name": "local"}, "object": {"key": f"incoming/{name}"}}}
                         for name in sys.argv[1:]]}
    t1 = time.time()
    out = aws_lambda.handler(event)
    print(f"init {t_init:.1f} s; cold start in handler: import {out['import_s']} s, "
          f"model load {out['model_load_s']} s; "
          f"handler {time.time() - t1:.1f} s for {len(sys.argv) - 1} scene(s); uid {os.getuid()}")
    written = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    print("written:", json.dumps(written, indent=1))
    leftovers = [str(p) for p in Path(aws_lambda.WORK).rglob("*") if p.is_file()]
    print("left in /tmp:", leftovers or "nothing")
    return 0 if out["results"] and all(r["status"] == "ok" for r in out["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
