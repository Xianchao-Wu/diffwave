#########################################################################
# File Name: 3.inference.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Jan 27 06:52:10 2023
#########################################################################
#!/bin/bash

model="/workspace/asr/diffusion_models/diffwave/checkpoint/weights.pt" # checkpoint
spec="/workspace/asr/diffusion_models/diffwave/data/A01M0097.wav_00440.050_00441.350.wav.spec.npy" # spectrogram
#spec="/workspace/asr/diffusion_models/diffwave/data/" # spectrogram

python -m ipdb inference.py \
    --fast \
    --model_dir $model \
    --spectrogram_path $spec \
    -o debug_output.wav

