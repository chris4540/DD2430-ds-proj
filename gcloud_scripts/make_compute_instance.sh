#!/bin/bash

# The name of the VM
export INSTANCE_NAME="chris-p4-dev"
# Opt to make the VM PREEMPTIBLE or not; Either "true" or "false"
# Doc: https://cloud.google.com/compute/docs/instances/preemptible
export IS_PREEMPTIBLE_MACHINE=false
# ------------------------------------------------
export ZONE="europe-west4-b"
export IMAGE_NAME="pytorch-1-3-rapids-0-10-cu100-20181117"
export INSTANCE_TYPE="n1-highmem-2"
export GPU_CONFIG="type=nvidia-tesla-p4,count=1"
export BOOT_DISK_SIZE="200GB"


args=(
  --machine-type=$INSTANCE_TYPE
  --zone=$ZONE
  --image=$IMAGE_NAME
  --maintenance-policy=TERMINATE
  --boot-disk-size=$BOOT_DISK_SIZE
  --accelerator=$GPU_CONFIG
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

