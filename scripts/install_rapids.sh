#!/bin/bash
#                           Script documentation
#  Target:
#     Install RAPIDS on the google clould compute image:
#       pytorch-1-3-cu100-20191112
#
#  Usage:
#     sudo bash ./install_cuml.sh
#
#  Notes:
#     1. The RAPIDS installation instruction page:
#        https://rapids.ai/start.html#rapids-release-selector
#     2. This script takes time. A installed image is created and named as:
#        pytorch-1-3-rapids-0-10-cu100-20181117
#     3. RAPIDS only works with NVIDIA Pascal™ GPU (i.e. either p4, p8, or p100)
#     4. The CUDA version of the image is "pytorch-1-3-cu100-20191112"
# ------------------------------------------------------------------------------

# exit when error
set -e

# Update the anaconda
conda install -y anaconda
# Install RAPIDS
conda install -y -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.10 python=3.7 cudatoolkit=10.1
