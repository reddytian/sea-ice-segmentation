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
START=$(date +%s)
aws s3 cp "$SCENE" "s3://$BUCKET/incoming/$NAME.nc" --region "$REGION"
echo -n "waiting for s3://$BUCKET/predictions/$NAME/ "
for _ in $(seq 1 90); do
  if aws s3 ls "s3://$BUCKET/predictions/$NAME/$NAME.json" --region "$REGION" >/dev/null 2>&1; then break; fi
  if aws s3 ls "s3://$BUCKET/predictions/$NAME/ERROR.json" --region "$REGION" >/dev/null 2>&1; then
    echo; aws s3 cp "s3://$BUCKET/predictions/$NAME/ERROR.json" - --region "$REGION"; exit 1
  fi
  echo -n "."; sleep 4
done
echo " $(( $(date +%s) - START )) s after upload started"
mkdir -p "predictions/aws/$NAME"
aws s3 cp --recursive "s3://$BUCKET/predictions/$NAME/" "predictions/aws/$NAME/" --region "$REGION"
python3 -c "import json, sys; m = json.load(open(sys.argv[1])); print('seconds in Lambda:', m['seconds'], '| model', m['model']['sha256'][:12])" \
  "predictions/aws/$NAME/$NAME.json"
echo "--- last Lambda log lines"
aws logs tail "/aws/lambda/$FUNCTION" --since 15m --region "$REGION" | grep -E "REPORT|results" | tail -4
