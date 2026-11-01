#!/usr/bin/env bash
# Remove everything deploy.sh created: bucket contents, the stack (function, bucket, role,
# logs, budget) and the ECR repository with its images. Nothing keeps billing afterwards.
#
#   ./deploy/aws/teardown.sh
set -euo pipefail

REGION=${AWS_REGION:-$(aws configure get region || true)}
REGION=${REGION:-us-east-2}
STACK=${STACK:-seaice-predict}
REPO=${REPO:-seaice-predict}
BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text 2>/dev/null || true)
if [[ -n "$BUCKET" && "$BUCKET" != None ]]; then
  aws s3 rm --recursive "s3://$BUCKET" --region "$REGION"     # CloudFormation cannot delete a non-empty bucket
fi
aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK" --region "$REGION"
aws ecr delete-repository --repository-name "$REPO" --region "$REGION" --force >/dev/null 2>&1 || true
echo "removed stack $STACK and ECR repository $REPO in $REGION"
