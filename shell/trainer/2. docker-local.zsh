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
image_name=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Run with local data as a mounted volume
docker run --rm \
  --volume $project_path/data/:/root/data/ \
  --name $container_name \
  $image_name \
    --data_dir /root/data/processed/time_intervals=1/resolution=5/ \
    --job_dir /root/train-output/ \
    --epochs 1