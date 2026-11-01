#!/usr/bin/env bash
# Build the Lambda image on this machine (linux/arm64), push it to ECR, deploy the stack.
#
#   ./deploy/aws/deploy.sh you@example.com
#
# Needs: Docker, AWS CLI v2 with credentials (aws configure). Region: $AWS_REGION, else
# the CLI's configured region, else us-east-2. Re-running redeploys the current commit.
set -euo pipefail
cd "$(dirname "$0")/../.."

EMAIL=${1:?usage: deploy/aws/deploy.sh ALERT_EMAIL}
REGION=${AWS_REGION:-$(aws configure get region || true)}
REGION=${REGION:-us-east-2}
STACK=${STACK:-seaice-predict}
REPO=${REPO:-seaice-predict}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
TAG="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src Dockerfile docker || echo -dirty)"
echo "account $ACCOUNT  region $REGION  stack $STACK  image $REPO:$TAG"

# --- ECR repository (scan on push; keep only the 3 newest images)
if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
fi
aws ecr put-lifecycle-policy --repository-name "$REPO" --region "$REGION" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep 3","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null

# --- build and push the `lambda` target
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/arm64 --target lambda --provenance=false -t "$REGISTRY/$REPO:$TAG" .
docker push "$REGISTRY/$REPO:$TAG"

# Lambda needs a single-platform manifest. If the push produced an image index, deploy the
# arm64 manifest inside it by digest.
read -r MTYPE DIGEST <<<"$(aws ecr batch-get-image --repository-name "$REPO" --region "$REGION" \
  --image-ids imageTag="$TAG" --query 'images[0].[imageManifestMediaType,imageId.imageDigest]' --output text)"
if [[ "$MTYPE" == *index* || "$MTYPE" == *list* ]]; then
  DIGEST=$(aws ecr batch-get-image --repository-name "$REPO" --region "$REGION" --image-ids imageTag="$TAG" \
    --query 'images[0].imageManifest' --output text | python3 -c '
import json, sys
m = json.load(sys.stdin)
print(next(x["digest"] for x in m["manifests"] if x.get("platform", {}).get("architecture") == "arm64"))')
fi
IMAGE="$REGISTRY/$REPO@$DIGEST"
echo "deploying $IMAGE"

# --- stack (a first deployment that failed must be deleted before retrying)
STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE)
if [[ "$STATUS" == ROLLBACK_COMPLETE ]]; then
  echo "previous first deployment failed; deleting it"
  aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK" --region "$REGION"
fi
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file deploy/aws/template.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides ImageUri="$IMAGE" AlertEmail="$EMAIL" --no-fail-on-empty-changeset
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs' --output table
echo "Confirm the AWS Budgets e-mail subscription if one arrives. Try it: deploy/aws/run_example.sh SCENE.nc"
