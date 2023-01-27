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
  nn.init.kaiming_normal_(layer.weight) # 这个是用kaiming方法来初始化网络
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps): # max_steps=50, 这是对时间的embedding
    super().__init__()
    self.register_buffer('embedding', 
            self._build_embedding(max_steps), persistent=False) 
    # [50, 128], used sin/cos for time embedding

    self.projection1 = Linear(128, 512) 
    # Linear(in_features=128, out_features=512, bias=True)

    self.projection2 = Linear(512, 512) 
    # Linear(in_features=512, out_features=512, bias=True)

  def forward(self, diffusion_step): 
    # timesteps, tensor([ 9, 30, 44, 25], device='cuda:0') of one batch
    #import ipdb; ipdb.set_trace()
    if diffusion_step.dtype in [torch.int32, torch.int64]: # NOTE, in here:
      x = self.embedding[diffusion_step] 
      # torch.Size([50, 128]), 这是事先设定好的一个可训练的table, 
      # [50, 128], now, x.shape=[4] to [4, 128]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x) 
    # Linear(in_features=128, out_features=512, bias=True), [4, 128] to [4, 512]
    # 第一个线性变换网络

    x = silu(x)
    x = self.projection2(x) 
    # Linear(in_features=512, out_features=512, bias=True), [4, 512] to [4, 512]
    # 第二个线性变换网络

    x = silu(x)
    return x # [4, 512], NOTE 这个关于时间的embedding，没有问题了

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps): 
    # max_steps=50, 这是关于时间的最大取值
    steps = torch.arange(max_steps).unsqueeze(1)  
    # [T,1], torch.Size([50, 1]), 0 to 49

    dims = torch.arange(64).unsqueeze(0)          
    # [1,64], torch.Size([1, 64]), tensor([[ 0,  1,  2, ..., 63]]) 

    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64], [50, 64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) 
    # 这里有sin, 也有cos, 所以是关于time的embedding了 NOTE, 这个代码写的很牛！！！

    return table # [50, 64*2]的一个embedding matrix，表格，供以后查询的时候使用


class SpectrogramUpsampler(nn.Module):
  # 梅尔谱的上采样 NOTE
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

  def forward(self, x): # torch.Size([4, 80, 62])
    import ipdb; ipdb.set_trace()
    x = torch.unsqueeze(x, 1) # torch.Size([4, 1, 80, 62])
    x = self.conv1(x) 
    # ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8)) 
    # -> torch.Size([4, 1, 80, 992])

    x = F.leaky_relu(x, 0.4) # 
    x = self.conv2(x) 
    # ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8)), 
    # -> torch.Size([4, 1, 80, 15872]) 
    # 这是对梅尔谱进行上采样，
    # 从而让其shape，和原始的wave form的shape相同 NOTE, 这个太重要了!!!

    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1) # torch.Size([4, 80, 15872])
    return x 
    # torch.Size([4, 80, 15872]) 这是把梅尔谱的shape，
    # 上采样成 original wave form的shape。然后就可以开始一系列操作了
    # NOTE from [1, 80, 82] to [1, 80, 20992] in inference

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
    # (torch.Size([4, 64, 15872]), 
    # torch.Size([4, 512]), torch.Size([4, 80, 15872]))
    #import ipdb; ipdb.set_trace()

    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) 
    # time, [4, 512] to torch.Size([4, 64, 1])
    # Linear(in_features=512, out_features=64, bias=True);  

    y = x + diffusion_step 
    # x=torch.Size([4, 64, 15872]), torch.Size([4, 64, 1]); 
    # TODO for what? x_t 和t 相加？？？y.shape = torch.Size([4, 64, 15872])
    # x_t和time的直接相加 NOTE 这个有意思！

    if self.conditioner_projection is None: # using a unconditional model
      y = self.dilated_conv(y)
    else:
      conditioner = self.conditioner_projection(conditioner) 
      # Conv1d(80, 128, kernel_size=(1,), stride=(1,)), 
      # torch.Size([4, 80, 15872]) -> torch.Size([4, 128, 15872]), 
      # 这是关于梅尔谱条件的input, condition

      y = self.dilated_conv(y) + conditioner
      # Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,)), 
      # from [4, 64, 15872] to [4, 128, 15872], 
      # NOTE，这是直接相加。。。 (x_t和t的相加) + (梅尔谱条件张量) 
      # y.shape = torch.Size([4, 128, 15872])

    gate, filter = torch.chunk(y, 2, dim=1) 
    # 按照维度dim=1，把y切成两个chunks，
    # gate.shape=[4, 64, 15872]; filter=[4, 64, 15872] NOTE

    y = torch.sigmoid(gate) * torch.tanh(filter) 
    # NOTE 有意思，左边的一半是给了sigmoid，右边的一半是给了filter 
    # TODO torch.Size([4, 64, 15872])=y.shape

    y = self.output_projection(y) 
    # Conv1d(64, 128, kernel_size=(1,), stride=(1,)); 
    # y=[4, 64, 15872] to torch.Size([4, 128, 15872])

    residual, skip = torch.chunk(y, 2, dim=1) 
    # residual.shape=[4, 64, 15872], skip.shape=[4, 64, 15872]

    return (x + residual) / sqrt(2.0), skip 
    # 左边[4, 64, 15872]; 右边[4, 64, 15872]


class DiffWave(nn.Module):
  def __init__(self, params):
    '''
    {'batch_size': 16, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 

    'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 

    'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}
    '''
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1) 
    # Conv1d(1, 64, kernel_size=(1,), stride=(1,))

    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule)) 
    # 50 -> TODO
    '''
    DiffusionEmbedding(
      (projection1): Linear(in_features=128, out_features=512, bias=True)
      (projection2): Linear(in_features=512, out_features=512, bias=True)
    )
    '''
    
    if self.params.unconditional: # use unconditional model
      self.spectrogram_upsampler = None
    else:
      self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels) 
      # params.n_mels=80 NOTE
      '''
      SpectrogramUpsampler(
          (conv1): ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
          (conv2): ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
        )
      '''

    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 
            2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers) 
        # 30 我勒个去，这是30层吗。。。so long... NOTE
    ])
    import ipdb; ipdb.set_trace()

    self.skip_projection = Conv1d(params.residual_channels, 
            params.residual_channels, 1)
    '''Conv1d(64, 64, kernel_size=(1,), stride=(1,))'''

    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    '''Conv1d(64, 1, kernel_size=(1,), stride=(1,))'''

    nn.init.zeros_(self.output_projection.weight)



  def forward(self, audio, diffusion_step, spectrogram=None): 
    # audio=[4, 15872], t=[9, 30, 44, 25], spectrogram=[4, 80, 62]
    # [inference], audio=[1, 20992], diffusion_step=tensor([0.], device='cuda:0'), spectrogram=torch.Size([1, 80, 82]) NOTE
    assert (spectrogram is None and self.spectrogram_upsampler is None) or \
           (spectrogram is not None and self.spectrogram_upsampler is not None)

    x = audio.unsqueeze(1) # [4, 1, 15872], 大概是1秒，因为1秒是16000个数据点 NOTE

    x = self.input_projection(x) 
    # Conv1d(1, 64, kernel_size=(1,), stride=(1,)), [4, 1, 15872] -> [4, 64, 15872]
    # [inference], [1, 1, 20992] -> [1, 64, 20992] NOTE
    x = F.relu(x)
    import ipdb; ipdb.set_trace()
    diffusion_step = self.diffusion_embedding(diffusion_step) 
    # from [4] to [4, 512], shape, 这是对时间time的编码，从一个标量，到一个512维度的向量 NOTE

    if self.spectrogram_upsampler: # use conditional model, NOTE in here
      spectrogram = self.spectrogram_upsampler(spectrogram) 
      # from [4, 80, 62] to [4, 80, 15872] 

    import ipdb; ipdb.set_trace()
    skip = None
    idx = 0
    for layer in self.residual_layers:
      print('---- {}-th layer ----'.format(idx))
      '''
      ResidualBlock(
          (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
          (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
          (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
          (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
        )
        '''
      print('{}-th layer, in: x={}, t={}, mel={}'.format(
          idx, x.shape, diffusion_step.shape, spectrogram.shape))

      x, skip_connection = layer(x, diffusion_step, spectrogram) 
      # input: (torch.Size([4, 64, 15872]), 
      # torch.Size([4, 512]), torch.Size([4, 80, 15872])); 
      # x_t, t, condition; output, x.shape=[4, 64, 15872], 
      # skip_connection=[4, 64, 15872] 

      print('{}-th layer, out: x={}, skip_connection={}'.format(
          idx, x.shape, skip_connection.shape))

      skip = skip_connection if skip is None else skip_connection + skip 
      # NOTE 这个有意思了

      idx += 1

    import ipdb; ipdb.set_trace()
    x = skip / sqrt(len(self.residual_layers)) # / sqrt(30), 是一个重要的分母...
    x = self.skip_projection(x) # Conv1d(64, 64, kernel_size=(1,), stride=(1,))
    x = F.relu(x)
    x = self.output_projection(x) # Conv1d(64, 1, kernel_size=(1,), stride=(1,))
    return x # e.g., [1, 64, 20992] -> output_projection -> torch.Size([1, 1, 20992]) -> in inference NOTE

    # 0-th, layer, x_t=[4, 64, 15872], t=[4, 512], mel=[4, 80, 15872]
    #              output: x=[4, 64, 15872], skip=[4, 64, 15872]


    '''
    ---- 0-th layer ----
    0-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    0-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 1-th layer ----
    1-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    1-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 2-th layer ----
    2-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    2-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 3-th layer ----
    3-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    3-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 4-th layer ----
    4-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    4-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 5-th layer ----
    5-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    5-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 6-th layer ----
    6-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    6-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 7-th layer ----
    7-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    7-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 8-th layer ----
    8-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    8-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 9-th layer ----
    9-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    9-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 10-th layer ----
    10-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    10-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 11-th layer ----
    11-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    11-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 12-th layer ----
    12-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    12-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 13-th layer ----
    13-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    13-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 14-th layer ----
    14-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    14-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 15-th layer ----
    15-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    15-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 16-th layer ----
    16-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    16-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 17-th layer ----
    17-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    17-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 18-th layer ----
    18-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    18-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 19-th layer ----
    19-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    19-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 20-th layer ----
    20-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    20-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 21-th layer ----
    21-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    21-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 22-th layer ----
    22-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    22-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 23-th layer ----
    23-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    23-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 24-th layer ----
    24-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    24-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 25-th layer ----
    25-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    25-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 26-th layer ----
    26-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    26-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 27-th layer ----
    27-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    27-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 28-th layer ----
    28-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    28-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    ---- 29-th layer ----
    29-th layer, in: x=torch.Size([4, 64, 15872]), t=torch.Size([4, 512]), mel=torch.Size([4, 80, 15872])
    29-th layer, out: x=torch.Size([4, 64, 15872]), skip_connection=torch.Size([4, 64, 15872])
    '''
