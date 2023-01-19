# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
T.set_audio_backend('sox_io')

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

#from diffwave.params import params
from params import params


def transform(filename):
    #import ipdb; ipdb.set_trace()
    if T.__version__ > '0.7.0':
        audio, sr = T.load(filename)
        audio = torch.clamp(audio[0], -1.0, 1.0)
    else:
      #audio, sr = T.load_wav(filename)
      audio, sr = T.load(filename)
      audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')
    mel_args = {
        'sample_rate': sr,
        'win_length': params.hop_samples * 4,
        'hop_length': params.hop_samples,
        'n_fft': params.n_fft,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': params.n_mels,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    with torch.no_grad():
        spectrogram = mel_spec_transform(audio) # e.g., from torch.Size([36368]) to torch.Size([80, 143])
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy()) # NOTE 有意思，这是保存梅尔谱的内容到一个具体的文件, /workspace/asr/diffusion_models/diffwave/data/A01M0097.wav_00442.278_00444.551.wav.spec.npy


def main(args):
    #import ipdb; ipdb.set_trace()
    filenames = glob(f'{args.dir}/**/*.wav', recursive=True)

    # TODO 下面的这个是多线程，现在先不用管
    #with ProcessPoolExecutor() as executor:
    #    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))

    for filename in filenames:
        transform(filename)


if __name__ == '__main__':
    #import ipdb; ipdb.set_trace()
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('dir',
        help='directory containing .wav files for training')
    main(parser.parse_args())

    ''' 
    root@95f4c42cafe7:/workspace/asr/diffusion_models/diffwave/data# ls -l
    total 396
    -rw-r--r-- 1 root root 41644 Jan 18 10:41 A01M0097.wav_00440.050_00441.350.wav
    -rw-r--r-- 1 root root 26368 Jan 18 11:00 A01M0097.wav_00440.050_00441.350.wav.spec.npy
    -rw-r--r-- 1 root root 72780 Jan 18 10:41 A01M0097.wav_00442.278_00444.551.wav
    -rw-r--r-- 1 root root 45888 Jan 18 11:00 A01M0097.wav_00442.278_00444.551.wav.spec.npy
    -rw-r--r-- 1 root root 85516 Jan 18 10:41 A01M0097.wav_00445.071_00447.742.wav
    -rw-r--r-- 1 root root 53568 Jan 18 11:00 A01M0097.wav_00445.071_00447.742.wav.spec.npy
    -rw-r--r-- 1 root root 38028 Jan 18 10:41 A01M0097.wav_00447.951_00449.138.wav
    -rw-r--r-- 1 root root 24128 Jan 18 11:00 A01M0097.wav_00447.951_00449.138.wav.spec.npy
    '''
