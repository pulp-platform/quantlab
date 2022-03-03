import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from typing import List, Dict

from .utils import SILENCE_LABEL
from .utils import GSCMapper, GSCPartition, GSCMeta


vocabulary = [  # task: GSCv2 - 2 + 10 words
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go'
]


def generate_preprocessing_parameters():

    # data pre-processing parameters
    sample_rate   = 16000   # Hertz (GSC-specific value)
    clip_duration = 1000.0  # ms
    window_size   = 40.0    # ms
    window_stride = 20.0    # ms
    time_shift    = 200.0   # ms

    clip_samples          = int((sample_rate * clip_duration) / 1000.0)
    window_size_samples   = int((sample_rate * window_size)   / 1000.0)
    window_stride_samples = int((sample_rate * window_stride) / 1000.0)
    time_shift_samples    = int((sample_rate * time_shift)    / 1000.0)

    # spectrogram
    clip_minus_one_window = (clip_samples - window_size_samples)
    spectrogram_length = 0 if (clip_minus_one_window < 0) else (1 + (clip_minus_one_window // window_stride_samples))
    feature_bin_count = 10  # for the spectrogram

    # for augmentation
    background_probability = 0.8
    background_volume      = 0.2

    parameters = {
        'sample_rate':            sample_rate,
        'clip_samples':           clip_samples,
        'window_size_samples':    window_size_samples,
        'window_stride_samples':  window_stride_samples,
        'time_shift_samples':     time_shift_samples,
        'spectrogram_length':     spectrogram_length,
        'feature_bin_count':      feature_bin_count,
        'background_probability': background_probability,
        'background_volume':      background_volume
    }

    return parameters


class GSCDataset(torch.utils.data.Dataset):

    def __init__(self,
                 partition: str,
                 path_data: os.PathLike):

        self._partition = partition
        self._gscmapper = GSCMapper(path_data=path_data, vocabulary=vocabulary)
        self._noise_waveforms = [self._load_waveform(gsc_file.path) for gsc_file in self._gscmapper.files_background]

    @property
    def word_to_index(self) -> Dict[str, int]:
        return self._gscmapper.word_to_index

    @property
    def noise_waveforms(self) -> List[torch.Tensor]:
        return self._noise_waveforms

    @property
    def dataset(self) -> List[GSCMeta]:
        return self._gscmapper.files[GSCPartition[self._partition.upper()].value]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):

        gsc_file = self.dataset[idx]

        spectrogram = self._waveform_to_spectrogram(gsc_file)
        label = self.word_to_index[gsc_file.word]

        return spectrogram, label

    @staticmethod
    def _load_waveform(wav_filepath: os.PathLike) -> torch.Tensor:
        waveform, _ = sf.read(wav_filepath)
        return torch.Tensor(np.array([waveform]))  # two-dimensional `torch.Tensor` of shape (1, n_samples)

    # TODO: all this function should be mapped to a proper pipeline of PyTorch transforms
    def _waveform_to_spectrogram(self, gsc_file: GSCMeta):
        
        parameters = generate_preprocessing_parameters()

        # load a waveform and ensure that it has the desired number of samples
        waveform = GSCDataset._load_waveform(gsc_file.path)
        missing_samples = parameters['clip_samples'] - waveform.shape[-1]
        if missing_samples < 0:  # truncate towards the end
            waveform = waveform[0][:parameters['clip_samples']].reshape(1, -1)
        elif missing_samples == 0:
            pass
        else:  # pad at the end
            waveform = F.pad(waveform, (0, parameters['clip_samples']), mode='constant', value=0.0)

        waveform = torch.mul(waveform, 0.0 if gsc_file.word == SILENCE_LABEL else 1.0)  # to "silence down" the arbitrarily-selected waveform (see `_finalise_file_lists` in `GSCMapper`)

        # first augmentation: shift the waveform along the time dimension
        if self._partition == GSCPartition.TRAIN.value:
            time_shift = np.random.randint(-parameters['time_shift_samples'], parameters['time_shift_samples']) if parameters['time_shift_samples'] > 0 else 0
            if time_shift < 0:  # pad at the end (utterance has already started "out" of the waveform)
                time_shift_padding = [0, -time_shift]
                time_shift_offset = -time_shift
            elif time_shift == 0:
                time_shift_padding = [0, 0]
                time_shift_offset = 0
            else:  # pad at the beginning (utterance starts later down the waveform)
                time_shift_padding = [time_shift, 0]
                time_shift_offset = 0

            waveform = F.pad(waveform, time_shift_padding, mode='constant', value=0.0)
            waveform = waveform[0][time_shift_offset:(time_shift_offset + parameters['clip_samples'])].reshape(1, -1)

        # second augmentation: generate additive background noise
        if (self._partition == GSCPartition.TRAIN.value) or (gsc_file.word == SILENCE_LABEL):

            # fetch random background noise waveform
            noise_waveform = random.choice(self._noise_waveforms)

            assert parameters['clip_samples'] < noise_waveform.shape[1]

            noise_offset = np.random.randint(0, noise_waveform.shape[1] - parameters['clip_samples'])
            noise_waveform = noise_waveform[0][noise_offset:(noise_offset + parameters['clip_samples'])].reshape(1, -1)
            noise_reshaped = noise_waveform.reshape([1, parameters['clip_samples']])

            if gsc_file.word == SILENCE_LABEL:
                noise_volume = np.random.uniform(0, 1)
            else:
                p = np.random.uniform(0, 1)
                noise_volume = np.random.uniform(0, 1) * (0.0 if p < parameters['background_probability'] else parameters['background_volume'])

            noise = torch.mul(noise_reshaped, noise_volume)
            waveform = torch.add(waveform, noise)

        # compute MFCCs (extract spectrogram)
        mfcc_transformation = torchaudio.transforms.MFCC(
            n_mfcc=parameters['feature_bin_count'],
            sample_rate=parameters['clip_samples'],
            melkwargs={
                'n_fft': 1024,
                'win_length': parameters['window_size_samples'],
                'hop_length': parameters['window_stride_samples'],
                'f_min': 20,
                'f_max': 4000,
                'n_mels': 40
            },
            log_mels=True,
            norm='ortho')
        spectrogram = mfcc_transformation(waveform)
        spectrogram = spectrogram[:, :, :parameters['spectrogram_length']].permute(0, 2, 1)
        spectrogram = torch.clip(spectrogram + 128.0, 0, 255.0)  # shift data in [0, 255] interval to match Dory request for uint8 inputs
        spectrogram = torch.floor(spectrogram)                   # integerise at training time, so fake-to-true conversion should be more unlikely to yield differences

        return torch.Tensor(spectrogram)
