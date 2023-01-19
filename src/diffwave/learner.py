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
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#from diffwave.dataset import from_path, from_gtzan
#from diffwave.model import DiffWave
#from diffwave.params import AttrDict

from dataset import from_path, from_gtzan
from model import DiffWave
from params import AttrDict


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    '''
        model_dir = '/workspace/asr/diffusion_models/diffwave/checkpoint'
        model = <class 'model.DiffWave'>
        dataset = <torch.utils.data.dataloader.DataLoader object at 0x7fe8d5300e50>
        optimizer = Adam 
        params = {'batch_size': 16, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}
        args = ()
        kwargs = {'fp16': False}
    '''
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False)) # <torch.cuda.amp.autocast_mode.autocast object at 0x7fe8dcd0df10>
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False)) # <torch.cuda.amp.grad_scaler.GradScaler object at 0x7fe8d51dc130>
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule) # (50,) from 0.0001 to 0.05
    noise_level = np.cumprod(1 - beta) # alpha_t_bar
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss() # L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device # device(type='cuda', index=0)
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    import ipdb; ipdb.set_trace()

    for param in self.model.parameters():
      param.grad = None

    audio = features['audio'] # torch.Size([4, 15872])
    spectrogram = features['spectrogram'] # torch.Size([4, 80, 62])
    import ipdb; ipdb.set_trace()
    N, T = audio.shape # N=4=batch size, T=15872=length of timepoints in wave form
    device = audio.device # device(type='cuda', index=0)
    self.noise_level = self.noise_level.to(device) # alpha_t_bar, 是1-beta的累计的乘积

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1) # [4,1], 这是从alpha_t_bar中，按照t，取了四个值出来，是为alpha_t_bar
      noise_scale_sqrt = noise_scale**0.5 # sqrt_alpha_t_bar
      noise = torch.randn_like(audio) # epsilon, [4, 15872]
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise # 这是一步炸楼，从x_0直接到x_t了。[4, 15872]

      predicted = self.model(noisy_audio, t, spectrogram) # x_t=[4, 15872], t=[4], condition=[4, 80, 62], 这仨输入有意思 NOTE, predicted.shape=[4, 1, 15872]
      loss = self.loss_fn(noise, predicted.squeeze(1)) # [4, 15872]=noise, and predicted_noise=[4, 15872]; L1Loss()=loss, tensor(0.7991, device='cuda:0', grad_fn=<L1LossBackward0>)=loss

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9) # tensor(0.2511, device='cuda:0')
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss # tensor(0.7991, device='cuda:0', grad_fn=<L1LossBackward0>) NOTE

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params): 
  '''
    replica_id = current gpu id = 0;
    model = <class 'model.DiffWave'>
    dataset = <torch.utils.data.dataloader.DataLoader object at 0x7fe8d5300e50>
    args = Namespace(data_dirs=['/workspace/asr/diffusion_models/diffwave/data'], fp16=False, max_steps=None, model_dir='/workspace/asr/diffusion_models/diffwave/checkpoint')
    params = {'batch_size': 16, 'learning_rate': 0.0002, 'max_grad_norm': None, 'sample_rate': 16000, 'n_mels': 80, 'n_fft': 1024, 'hop_samples': 256, 'crop_mel_frames': 62, 'residual_layers': 30, 'residual_channels': 64, 'dilation_cycle_length': 10, 'unconditional': False, 'noise_schedule': [0.0001, 0.0011183673469387756, 0.002136734693877551, 0.0031551020408163264, 0.004173469387755102, 0.005191836734693878, 0.006210204081632653, 0.007228571428571429, 0.008246938775510203, 0.009265306122448979, 0.010283673469387754, 0.01130204081632653, 0.012320408163265305, 0.013338775510204081, 0.014357142857142857, 0.015375510204081632, 0.016393877551020408, 0.017412244897959183, 0.01843061224489796, 0.019448979591836734, 0.02046734693877551, 0.021485714285714285, 0.02250408163265306, 0.023522448979591836, 0.02454081632653061, 0.025559183673469387, 0.026577551020408163, 0.027595918367346938, 0.028614285714285714, 0.02963265306122449, 0.030651020408163265, 0.031669387755102044, 0.03268775510204082, 0.033706122448979595, 0.03472448979591837, 0.035742857142857146, 0.03676122448979592, 0.0377795918367347, 0.03879795918367347, 0.03981632653061225, 0.04083469387755102, 0.0418530612244898, 0.042871428571428574, 0.04388979591836735, 0.044908163265306125, 0.0459265306122449, 0.046944897959183676, 0.04796326530612245, 0.04898163265306123, 0.05], 'inference_noise_schedule': [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], 'audio_len': 80000}
  '''
  import ipdb; ipdb.set_trace()
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
  '''
  Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.0002
        weight_decay: 0
    )
  '''
  import ipdb; ipdb.set_trace()
  learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  # <learner.DiffWaveLearner object at 0x7fe8d2dfec40>

  learner.is_master = (replica_id == 0) # True
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps) # args.max_steps = None


def train(args, params):
  import ipdb; ipdb.set_trace()

  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else: # NOTE in here:
    dataset = from_path(args.data_dirs, params) # <torch.utils.data.dataloader.DataLoader object at 0x7f32b028feb0>
  model = DiffWave(params).cuda()
  '''
  ipdb> p model
DiffWave(
  (input_projection): Conv1d(1, 64, kernel_size=(1,), stride=(1,))
  (diffusion_embedding): DiffusionEmbedding(
    (projection1): Linear(in_features=128, out_features=512, bias=True)
    (projection2): Linear(in_features=512, out_features=512, bias=True)
  )
  (spectrogram_upsampler): SpectrogramUpsampler(
    (conv1): ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
    (conv2): ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16), padding=(1, 8))
  )
  (residual_layers): ModuleList(
    (0): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (1): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (2): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (3): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (4): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(16,), dilation=(16,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (5): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(32,), dilation=(32,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (6): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(64,), dilation=(64,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (7): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(128,), dilation=(128,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (8): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(256,), dilation=(256,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (9): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(512,), dilation=(512,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (10): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (11): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (12): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (13): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (14): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(16,), dilation=(16,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (15): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(32,), dilation=(32,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (16): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(64,), dilation=(64,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (17): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(128,), dilation=(128,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (18): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(256,), dilation=(256,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (19): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(512,), dilation=(512,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (20): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (21): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (22): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (23): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (24): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(16,), dilation=(16,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (25): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(32,), dilation=(32,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (26): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(64,), dilation=(64,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (27): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(128,), dilation=(128,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (28): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(256,), dilation=(256,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
    (29): ResidualBlock(
      (dilated_conv): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(512,), dilation=(512,))
      (diffusion_projection): Linear(in_features=512, out_features=64, bias=True)
      (conditioner_projection): Conv1d(80, 128, kernel_size=(1,), stride=(1,))
      (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
    )
  )
  (skip_projection): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
  (output_projection): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
)
'''


  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  import ipdb; ipdb.set_trace()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params)
