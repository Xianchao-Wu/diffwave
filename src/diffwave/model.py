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
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps): # max_steps=50
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False) # [50, 128], used sin/cos for time embedding
    self.projection1 = Linear(128, 512) # Linear(in_features=128, out_features=512, bias=True)
    self.projection2 = Linear(512, 512) # Linear(in_features=512, out_features=512, bias=True)

  def forward(self, diffusion_step):
    import ipdb; ipdb.set_trace()
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps): # max_steps=50, 这是关于时间的最大取值
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1], torch.Size([50, 1]), 0 to 49
    dims = torch.arange(64).unsqueeze(0)          # [1,64], torch.Size([1, 64]), tensor([[ 0,  1,  2, ..., 63]]) 
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64], [50, 64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) # 这里有sin, 也有cos, 所以是关于time的embedding了 NOTE, 这个代码写的很牛！！！
    return table # [50, 64*2]的一个embedding matrix，表格，供以后查询的时候使用


class SpectrogramUpsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8]) 
    '''
    ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
    '''
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])
    '''
    ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
    '''

  def forward(self, x):
    import ipdb; ipdb.set_trace()
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=False):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional; 80
    :param residual_channels: audio conv; 64
    :param dilation: audio conv dilation; 1
    :param uncond: disable spectrogram conditional; False
    '''
    #import ipdb; ipdb.set_trace()
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 
            3, padding=dilation, dilation=dilation)
    '''Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))'''

    self.diffusion_projection = Linear(512, residual_channels) 
    '''Linear(in_features=512, out_features=64, bias=True)'''

    if not uncond: # conditional model
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        '''Conv1d(80, 128, kernel_size=(1,), stride=(1,))'''
    else: # unconditional model
        self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
    '''Conv1d(64, 128, kernel_size=(1,), stride=(1,))'''

  def forward(self, x, diffusion_step, conditioner=None):
    import ipdb; ipdb.set_trace()

    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    if self.conditioner_projection is None: # using a unconditional model
      y = self.dilated_conv(y)
    else:
      conditioner = self.conditioner_projection(conditioner)
      y = self.dilated_conv(y) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, params):
    '''
    {'batch_size': 16, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 

    'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 

    'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}
    '''
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1) # Conv1d(1, 64, kernel_size=(1,), stride=(1,))
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule)) # 50 -> TODO
    '''
    DiffusionEmbedding(
      (projection1): Linear(in_features=128, out_features=512, bias=True)
      (projection2): Linear(in_features=512, out_features=512, bias=True)
    )
    '''
    
    if self.params.unconditional: # use unconditional model
      self.spectrogram_upsampler = None
    else:
      self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels) # params.n_mels=80 NOTE
      '''
      SpectrogramUpsampler(
          (conv1): ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
          (conv2): ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
        )
      '''

    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 
            2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers) # 30 我勒个去，这是30层吗。。。so long... NOTE
    ])
    import ipdb; ipdb.set_trace()

    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    '''Conv1d(64, 64, kernel_size=(1,), stride=(1,))'''
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    '''Conv1d(64, 1, kernel_size=(1,), stride=(1,))'''
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, spectrogram=None):
    assert (spectrogram is None and self.spectrogram_upsampler is None) or \
           (spectrogram is not None and self.spectrogram_upsampler is not None)
    x = audio.unsqueeze(1)
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    if self.spectrogram_upsampler: # use conditional model
      spectrogram = self.spectrogram_upsampler(spectrogram)

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, spectrogram)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x
