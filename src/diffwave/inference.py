# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser

#from diffwave.params import AttrDict, params as base_params
#from diffwave.model import DiffWave

from params import AttrDict, params as base_params
from model import DiffWave


models = {}

def predict(spectrogram=None, model_dir=None, # spectrogram.shape=[80, 82] 
        params=None, device=torch.device('cuda'), fast_sampling=False): # 
  # Lazy load model.
  import ipdb; ipdb.set_trace()
  if not model_dir in models: # models={}
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir) # NOTE here, loaded okay
    model = DiffWave(AttrDict(base_params)).to(device) # {'batch_size': 4, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  import ipdb; ipdb.set_trace()
  model = models[model_dir] # 这是个词典,key="/workspace/asr/diffusion_models/diffwave/checkpoint/weights.pt", value=model object
  model.params.override(params) # NOTE TODO 用override是干啥的?
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule) # beta = noise_schedule=[0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05]
    inference_noise_schedule = np.array(
            model.params.inference_noise_schedule # NOTE [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5] 设定了fast，所以betas的实际取值是这个list。
        ) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule # alpha_t 原来的
    talpha_cum = np.cumprod(talpha) # alpha_t_bar 积累的

    beta = inference_noise_schedule # beta，短的array([1.e-04, 1.e-03, 1.e-02, 5.e-02, 2.e-01, 5.e-01])
    alpha = 1 - beta # alpha, array([0.9999, 0.999 , 0.99  , 0.95  , 0.8   , 0.5   ])
    alpha_cum = np.cumprod(alpha) # array([0.9999    , 0.9989001 , 0.9889111 , 0.93946554, 0.75157244, 0.37578622])

    T = []
    for s in range(len(inference_noise_schedule)): # beta, 6
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - 
              alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32) # array([ 0.       ,  0.8941341,  4.086654 , 10.451817 , 22.992493 , 42.918644 ], dtype=float32)

    import ipdb; ipdb.set_trace()
    if not model.params.unconditional:
      if len(spectrogram.shape) == 2: # NOTE here
        # Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0) # torch.Size([1, 80, 82])
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], 
              model.params.hop_samples * spectrogram.shape[-1], device=device) # [1, 256*82], (batch, seq_length); audio.shape=torch.Size([1, 20992]) NOTE
    else:
      audio = torch.randn(1, params.audio_len, device=device) # not here
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    # array([0.9999    , 0.9989001 , 0.9889111 , 0.93946554, 0.75157244, 0.37578622]) -> array([0.99995   , 0.9994499 , 0.99444009, 0.96926031, 0.86693277,  0.61301404])
    import ipdb; ipdb.set_trace()
    for n in range(len(alpha) - 1, -1, -1): # [5, 4, 3, 2, 1, 0]
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - 
                    c2 * model(audio, # torch.Size([1, 20992])
                         torch.tensor([T[n]], # array([ 0.       ,  0.8941341,  4.086654 , 10.451817 , 22.992493 ,        42.918644 ], dtype=float32) --> NOTE
                         device=audio.device), 
                         spectrogram).squeeze(1)) # torch.Size([1, 80, 82]) NOTE model.output=[1, 1, 20992] NOTE
      if n > 0:
        noise = torch.randn_like(audio) # audio.shape=[1, 20992]
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0) # audio.shape=[1, 20992]

    import ipdb; ipdb.set_trace()

  return audio, model.params.sample_rate # audio.shape=[1, 20992], sample_rate=16000=16K


def main(args):
  import ipdb; ipdb.set_trace()
  if args.spectrogram_path: # /workspace/asr/diffusion_models/diffwave/data/A01M0097.wav_00440.050_00441.350.wav.spec.npy
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path)) # torch.Size([80, 82]), 80=梅尔谱的维度，82=音频的长度
  else:
    spectrogram = None
  audio, sr = predict(spectrogram, 
    model_dir=args.model_dir, fast_sampling=args.fast, params=base_params) # model_dir="'/workspace/asr/diffusion_models/diffwave/checkpoint/weights.pt'", args.fast=True, params={'batch_size': 4, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}

  import ipdb; ipdb.set_trace()
  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(
    description='runs inference on a spectrogram file' + 
        'generated by diffwave.preprocess')

  parser.add_argument('--model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')

  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')

  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')

  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')

  import ipdb; ipdb.set_trace()
  main(parser.parse_args())

