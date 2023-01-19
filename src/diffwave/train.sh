#########################################################################
# File Name: train.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Jan 18 11:06:30 2023
#########################################################################
#!/bin/bash

ckpt="/workspace/asr/diffusion_models/diffwave/checkpoint"
data="/workspace/asr/diffusion_models/diffwave/data"

python -m ipdb __main__.py $ckpt $data
