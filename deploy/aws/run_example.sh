#!/usr/bin/env bash
# Upload one scene, wait for the Lambda to write the maps, download them, show timings.
#
#   ./deploy/aws/run_example.sh ~/dl-local/ai4arctic/scenes/20210421T082208_dmi_prep.nc
set -euo pipefail
cd "$(dirname "$0")/../.."

SCENE=${1:?usage: deploy/aws/run_example.sh SCENE.nc}
REGION=${AWS_REGION:-$(aws configure get region || true)}
REGION=${REGION:-us-east-2}
STACK=${STACK:-seaice-predict}
out() { aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
          --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
BUCKET=$(out BucketName)
FUNCTION=$(out FunctionName)
NAME=$(basename "$SCENE" .nc)

aws s3 rm --recursive "s3://$BUCKET/predictions/$NAME/" --region "$REGION" >/dev/null
T0=$(date +%s)
T0_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)          # logs from this run only
aws s3 cp "$SCENE" "s3://$BUCKET/incoming/$NAME.nc" --region "$REGION"
START=$(date +%s)
echo "upload took $(( START - T0 )) s (your connection, not the pipeline)"
echo -n "waiting for s3://$BUCKET/predictions/$NAME/ "
for _ in $(seq 1 90); do
  if aws s3 ls "s3://$BUCKET/predictions/$NAME/$NAME.json" --region "$REGION" >/dev/null 2>&1; then break; fi
  if aws s3 ls "s3://$BUCKET/predictions/$NAME/ERROR.json" --region "$REGION" >/dev/null 2>&1; then
    echo; aws s3 cp "s3://$BUCKET/predictions/$NAME/ERROR.json" - --region "$REGION"; exit 1
  fi
  echo -n "."; sleep 4
done
echo " $(( $(date +%s) - START )) s from upload to maps in S3"
mkdir -p "predictions/aws/$NAME"
aws s3 cp --recursive "s3://$BUCKET/predictions/$NAME/" "predictions/aws/$NAME/" --region "$REGION"
python3 -c "import json, sys; m = json.load(open(sys.argv[1])); print('seconds in Lambda:', m['seconds'], '| model', m['model']['sha256'][:12])" \
  "predictions/aws/$NAME/$NAME.json"
# The REPORT line (duration, billed time, max memory used) is written a few seconds after
# the handler returns, i.e. after the maps are already in S3.
echo "--- Lambda log for this run"
LOG=""
for _ in $(seq 1 15); do
  LOG=$(aws logs tail "/aws/lambda/$FUNCTION" --since "$T0_ISO" --region "$REGION" \
          | grep -E "INIT_REPORT|REPORT RequestId|\"results\"" || true)
  if grep -q "REPORT RequestId" <<<"$LOG"; then break; fi
  sleep 3
done
echo "${LOG:-no REPORT line yet; try: aws logs tail /aws/lambda/$FUNCTION --since 10m --region $REGION}"
