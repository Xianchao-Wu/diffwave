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
import random
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class ConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    ##import ipdb; ipdb.set_trace()

    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True) # 这是获取所有的wav文件，嗯？梅尔谱文件被直接无视了吗... NOTE TODO

  def __len__(self):
    #import ipdb; ipdb.set_trace()
    return len(self.filenames) # =4

  def __getitem__(self, idx):
    #import ipdb; ipdb.set_trace()
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load(audio_filename) # NOTE 哦，明白了，这是根据wav文件名，去找的 wav.spec.npy这个保存了梅尔谱的文件名了, shape = torch.Size([1, 42736])
    spectrogram = np.load(spec_filename) # shape = (80, 167)
    return {
        'audio': signal[0], # torch.Size([42736]) NOTE
        'spectrogram': spectrogram.T # (167, 80)
    }


class UnconditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    #import ipdb; ipdb.set_trace()

    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load(audio_filename)
    return {
        'audio': signal[0],
        'spectrogram': None
    }


class Collator:
  def __init__(self, params):
    self.params = params
    '''
    {'batch_size': 16, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}
    '''

  def collate(self, minibatch): # minibatch = a list of dicts, each element is a dict, {'audio': tensor, 'spectrogram': array}, collate = 检查校对 的意思
    #import ipdb; ipdb.set_trace()
    samples_per_frame = self.params.hop_samples # 256 NOTE
    for record in minibatch:
      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          end = start + self.params.audio_len
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      else:
          # Filter out records that aren't long enough. NOTE in here
          if len(record['spectrogram']) < self.params.crop_mel_frames: # self.params.crop_mel_frames = 62，如果长度不超过62个梅尔frame，那当前的这个样本，就不要了
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames) # 截取，起点, e.g., 76
          end = start + self.params.crop_mel_frames # 截取，终点, e.g., 138, (开区间，只到137即可)
          record['spectrogram'] = record['spectrogram'][start:end].T # NOTE, 如果长度超过了62，那么只需要保留62的长度就可以了.

          start *= samples_per_frame # 256，大意是，一个梅尔谱frame里面，对应了256个采样点。 =76*256=19456
          end *= samples_per_frame # =138*256=35328
          record['audio'] = record['audio'][start:end] # torch.Size([15872]), 这是截取了一部分的原始wave form里面的数据
          record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant') # 这里什么都没有做
    #import ipdb; ipdb.set_trace()
    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    if self.params.unconditional:
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': None,
        }
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio), # [4, 15872]
        'spectrogram': torch.from_numpy(spectrogram), # (4, 80, 62)
    }

  # for gtzan
  def collate_gtzan(self, minibatch):
    ldata = []
    mean_audio_len = self.params.audio_len # change to fit in gpu memory
    # audio total generated time = audio_len * sample_rate
    # GTZAN statistics
    # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
    # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
    # sample rate of all is 22050
    for data in minibatch:
      if data[0].shape[-1] < mean_audio_len:  # pad
        data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
      elif data[0].shape[-1] > mean_audio_len:  # crop
        start = random.randint(0, data[0].shape[-1] - mean_audio_len)
        end = start + mean_audio_len
        data_audio = data[0][:, start:end]
      else:
        data_audio = data[0]
      ldata.append(data_audio)
    audio = torch.cat(ldata, dim=0)
    return {
          'audio': audio,
          'spectrogram': None,
    }


def from_path(data_dirs, params, is_distributed=False):
  #import ipdb; ipdb.set_trace()

  if params.unconditional:
    dataset = UnconditionalDataset(data_dirs)
  else:#with condition NOTE, here in
    dataset = ConditionalDataset(data_dirs) # ['/workspace/asr/diffusion_models/diffwave/data']
  return torch.utils.data.DataLoader(
      dataset, # <dataset.ConditionalDataset object at 0x7fe8e4362940>
      batch_size=params.batch_size, # 16
      collate_fn=Collator(params).collate, # 
      shuffle=not is_distributed, # is_distributed=false
      num_workers=0, #os.cpu_count(), # os.cpu_count()=80 啊，有钱，，，居然有80个内核... NOTE
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=False) # TODO True to False


def from_gtzan(params, is_distributed=False):
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
