#!/bin/bash
#                           Script documentation
#  Target:
#     Download the img.zip in the:
#       Category and Attribute Prediction Benchmark of DeepFashion
#
#  Usage:
#     ./download_deepfashion_ds.sh
#
#  Dataset webpage:
#     http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
#
#  Notes:
#     1. The img.zip shared link is:
#        https://drive.google.com/open?id=0B7EVK8r0v71pa2EyNEJ0dE9zbU0
#        Therefore the google drive file id is: "0B7EVK8r0v71pa2EyNEJ0dE9zbU0"
#     2. The dataset contains other text file, but they are included in the repo.
# ------------------------------------------------------------------------------

# exit when error
set -e

# Config
IMAGE_DATA_ID=0B7EVK8r0v71pa2EyNEJ0dE9zbU0
MD5_SUM=48aca947353f2235d27a52e6707205b7

# Check and install gdown
if [[ ! -x "$(command -v gdown)" ]]; then
    echo "Please install gdown by the following command."
    echo "pip install gdown"
    exit 1
fi
# ------------------------------------------------------------------------------
# Get the script dir first
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Get the data dir
data_dir=${script_dir}/../deepfashion_data
mkdir -p ${data_dir}

# 0. remove any existing residual
rm -f img.zip

# 1. download the dataset (img.zip)
gdown --id ${IMAGE_DATA_ID} -O img.zip

# 2. check md5 sum
echo "Checking MD5 sum..."
echo "${MD5_SUM} *img.zip" | md5sum -c

# 3. move to data_dir
mv img.zip ${data_dir}

# 4. Unzip the image folder
cd ${data_dir}
	unzip img.zip
cd -

