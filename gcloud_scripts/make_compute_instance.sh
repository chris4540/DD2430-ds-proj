#!/bin/bash

# The name of the VM
export INSTANCE_NAME="tch13-p100"
export ZONE="europe-west1-d"
# Opt to make the VM PREEMPTIBLE or not; Either "true" or "false"
# Doc: https://cloud.google.com/compute/docs/instances/preemptible
export IS_PREEMPTIBLE_MACHINE=false
# ------------------------------------------------
export IMAGE_NAME="pytorch-1-3-rapids-0-10-cu100-20181117"
export INSTANCE_TYPE="n1-standard-8"
export GPU_CONFIG="type=nvidia-tesla-p100,count=1"
export BOOT_DISK_SIZE="200GB"
export DISK_NAME=tch13-p100-preep


args=(
  --machine-type=$INSTANCE_TYPE
  --zone=$ZONE
  # --image=$IMAGE_NAME
  --maintenance-policy=TERMINATE
  # --boot-disk-size=$BOOT_DISK_SIZE
  --accelerator=$GPU_CONFIG
  --disk="name=${DISK_NAME},boot=yes,mode=rw"
  --metadata="install-nvidia-driver=True"
)
if ${IS_PREEMPTIBLE_MACHINE}; then
  args+=( --preemptible )
fi

# print the create vm options out
echo "============================================="
echo VM instance name: "${INSTANCE_NAME}"
echo "--------------------------------------"
echo Create VM options:
for opt in "${args[@]}"; do
  echo ${opt}
done
echo "============================================="

echo "Creating VM...."
gcloud compute instances create ${INSTANCE_NAME} "${args[@]}"

