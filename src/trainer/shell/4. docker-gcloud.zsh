#!/bin/zsh
script_dir=$(dirname $0:A);
package_path=$(dirname $script_dir)
package_name=$(basename $package_path);
project_path=$(dirname $(dirname $package_path));
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


# Parse JSON config file
config_list=$(cat $script_dir/configs.json)
last_id=$(echo $config_list | jq ".[-1].id?")
config_id=$last_id
echo

# Parse arguments
while [[ $# -gt 0 ]]
do
    case $1 in
    -i|--id)
        # Store config ID - if argument is numeric and the config ID exists
        if [[ $2 =~ ^[0-9]+$ ]] && [ $2 -le $last_id ] && [ $2 -ge 0 ]; then
            config_id=$2
            shift 2
        else
            echo "Error: No config found with id $2." && return 1
        fi
        ;;
    -r|--rebuild)
        # Rebuild image
        ( cd $project_path && . docker/$package_name/docker-build.zsh ) || return 1
        shift
        ;;
    -p|--push)
        # Push image
        docker push $IMAGE_URI
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Parse config
echo "Using training config id $config_id:"
config=$(echo $config_list | jq ".[$config_id].config?")
echo $config
data_id=$(echo $config_list | jq ".[$config_id].config[data_id]?")
echo "Applied to data preprocessed with config $data_id"
echo


# Data paths
data_path=data/processed/spectrograms/config-$data_id/nsynth-train
output_path=output/${JOB_NAME}

# Submit training job to gcloud AI platform
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
    --data_dir=gs://$BUCKET_NAME/$data_path \
    --job_dir=gs://$BUCKET_NAME/$output_path \
    --config $config \
    --epochs=100