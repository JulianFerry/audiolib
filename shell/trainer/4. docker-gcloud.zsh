#!/bin/zsh
script_dir=$(dirname $0:A);
package_name=$(basename $script_dir);
project_path=$(dirname $(dirname $script_dir));
project_name=$(basename $project_path);
container_name=$project_name-$package_name;

# GCP AI platform container naming
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=$container_name
IMAGE_TAG=latest

export REGION=europe-west1
export BUCKET_NAME=deep-musik-data
export IMAGE_URI=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export JOB_NAME=${package_name}_$(date +%Y%m%d_%H%M%S);

# Rebuild and/or push image if args are set
# Rebuild image if arg is set
for arg in $@
do
  case $arg in
    -r|--rebuild)
      ( cd $project_path && . docker/$package_name/docker-build.zsh );;
    -p|--push)
      docker push $IMAGE_URI
  esac
done

# Submit training job to gcloud AI platform
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
    --data_dir=gs://$BUCKET_NAME/data/processed/time_intervals=1/resolution=5/ \
    --job_dir=gs://$BUCKET_NAME/train-output/$JOB_NAME \
    --instruments "[keyboard_acoustic, guitar_acoustic]" \
    --epochs=10