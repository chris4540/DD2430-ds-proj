#!/bin/bash

export ZONE="us-west1-b"
export IMAGE_NAME="pytorch-1-3-cu100-20191112"
export INSTANCE_NAME="torch13-chris-k80"
export INSTANCE_TYPE="n1-highmem-2"
export GPU_CONFIG="type=nvidia-tesla-k80,count=1"
export BOOT_DISK_SIZE="100GB"

gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$INSTANCE_TYPE \
  --zone=$ZONE \
  --image=$IMAGE_NAME \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --maintenance-policy=TERMINATE \
  --accelerator=$GPU_CONFIG \
  --metadata="install-nvidia-driver=True"
