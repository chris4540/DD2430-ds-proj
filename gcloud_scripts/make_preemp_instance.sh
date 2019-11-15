#!/bin/bash

export ZONE="europe-west4-b"
export IMAGE_NAME="pytorch-1-3-cu100-notebooks-20191112"
export INSTANCE_NAME="torch13-chris-p4-premp"
export INSTANCE_TYPE="n1-highmem-2"
export GPU_CONFIG="type=nvidia-tesla-p4,count=1"
export BOOT_DISK_SIZE="100GB"

gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$INSTANCE_TYPE \
  --zone=$ZONE \
  --image=$IMAGE_NAME \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --accelerator=$GPU_CONFIG \
  --no-boot-disk-auto-delete \
  --metadata="install-nvidia-driver=True" \
  --preemptible
