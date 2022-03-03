#
# utils.py
#
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
#
# Copyright (c) 2020-2022 ETH Zuerich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
import hashlib
import random
import math
from enum import unique, Enum, auto

from typing import Dict, List, NamedTuple


MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
RANDOM_SEED = 59185

BACKGROUND_NOISE_LABEL = '_background_noise_'

SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0

UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1


CLASSES = [  # _, CLASSES, _ = next(iter(os.walk('/path/to/data')))
    '_background_noise_',
    # nature
    'bird',
    'cat',
    'dog',
    'tree',
    # humans
    'house',
    'bed',
    'happy',
    'wow',
    'sheila',
    'marvin',
    'visual',
    # imperative commands
    'yes',
    'no',
    'on',
    'off',
    'learn',
    # movement commands
    'go',
    'stop',
    'up',
    'down',
    'forward',
    'backward',
    'left',
    'right',
    'follow',
    # numbers
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
]


# As a first capability, I need a data structure to represent WAV files. Each
# file in the Google Speech Command data set has the following properties:
#
#   * it has a unique path descending to it from the common root (i.e., from
#     the directory containing the unpacked data set);
#   * it encodes the utterance of a specific word (which is represented by the
#     name of its direct enclosing directory);
#   * it has been uttered by a specific speaker, which identity is anonymised
#     in the form of an eight-digit hexadecimal number.

class GSCMeta(NamedTuple):
    path: os.PathLike
    word: str
    speaker_id: str


def get_wav_file_metadata(wav_path: os.PathLike) -> GSCMeta:
    """Infer GSC-specific information."""

    # decompose path to extract file metadata
    wav_path_dir, wav_file_filename = os.path.split(wav_path)

    # extract file metadata
    _, word = os.path.split(wav_path_dir)
    speaker_id = '' if word == BACKGROUND_NOISE_LABEL else wav_file_filename.split('_')[0]  # anonymised speaker identity (eight-digit hexadecimal number)

    return GSCMeta(path=wav_path, word=word, speaker_id=speaker_id)


# Google Speech Commands does not come with a pre-defined partition of its
# data points into training, validation, and test partitions. For scientific
# replicability reasons, we need to generate such a partition in a way that
# respects three properties:
#
#   * the partition must be deterministic across different instantiations of
#     the data set;
#   * each utterance class (i.e., each word) is represented in a balanced way
#     across all three partitions;
#   * secondary variability factors (the word class being the primary one)
#     should not cross the boundaries between the three partitions; indeed,
#     machine learning models might adapt to secondary variability factors
#     that are equally represented in all the three elements of the partition,
#     conveying the false impression that the model is adapting to the primary
#     variability factor; in the case of GSC, this secondary variability
#     factor is the speaker's identity.

@unique
class GSCPartition(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class GSCPartitioner(object):

    def __init__(self, test_fraction: float = 0.1, valid_fraction: float = 0.1):

        if not (0.0 <= test_fraction <= 1.0):
            raise ValueError(f"The fraction of data points that should be directed to the test partition should be a number in [0, 1], but received {valid_fraction}.")

        if not (0.0 <= valid_fraction <= 1.0):
            raise ValueError(f"The fraction of data points that should be directed to the validation partition should be a number in [0, 1], but received {valid_fraction}.")

        if 1.0 == (test_fraction + valid_fraction):
            raise Warning(f"The fraction of data points that should be directed to the training partition is zero.")
        elif 1.0 < (test_fraction + valid_fraction):
            raise ValueError(f"The total fraction of data points that should be directed to the test and validation partitions can not exceed one.")

        self._test_fraction = test_fraction
        self._valid_fraction = valid_fraction

    def which_set(self, gsc_file: GSCMeta) -> GSCPartition:
        # We need to decide whether this GSC file should go into the training,
        # validation, or testing set. The selection should be conditioned
        # deterministically on the speaker's identically (so that adding new
        # files later will preserve the assignments for the instances from
        # those speakers which are already represented), but should also have
        # a probability to end up in the test or validation set that is
        # proportional to `self._test_fraction` and `self._valid_fraction`,
        # respectively. We achieve this by hashing the speaker's ID and
        # mapping the resulting integer to a number in [0, 1].

        hash_name_hashed = hashlib.sha1(gsc_file.speaker_id.encode()).hexdigest()
        fraction = (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) / float(MAX_NUM_WAVS_PER_CLASS)  # remember: N % K is an integer going from 0 to K - 1

        # if fraction < self._test_fraction:
        #     set_ = GSCPartition.TEST.value
        # elif fraction < (self._test_fraction + self._valid_fraction):
        #     set_ = GSCPartition.VALID.value
        # TODO: legacy
        if fraction < self._valid_fraction:
            set_ = GSCPartition.VALID.value
        elif fraction < (self._valid_fraction + self._test_fraction):
            set_ = GSCPartition.TEST.value
        else:
            set_ = GSCPartition.TRAIN.value

        return set_


# Now I need to assemble the data sets.
# There are four "class types" of classes:
#
#   * the special `_background_noise_` class, which is represented by a
#     dedicated GSC sub-folder;
#   * "vocabulary" classes, i.e., those GSC classes which have a dedicated
#     sub-folder and are specified as `target_words`;
#   * "unknown" class, i.e., the union of those GSC classes which have a
#     dedicated sub-folder but are not specified as `target_words`;
#   * "silence" class; waveforms from this class can be obtained by simply
#     zeroing out a given waveform.

class GSCMapper(object):
    """A class mapping GSC files to a training/validation/test partition.

    The purpose of this class is defining a mapping of Google Speech Command
    files to the sets of the training/validation/test partition. We do not
    want to produce static indices of such partitions (e.g., text files)
    because it is nice to have the possibility of dynamically including or
    excluding certain words to develop different applications.

    """
    # TODO: this use case shows that we might want to pass keyword arguments to the constructors of PyTorch `Dataset`s
    def __init__(self,
                 path_data: os.PathLike,
                 vocabulary: List[str],
                 test_fraction: float = 0.1,   # these describe the proportion of GSC data points that should end up in the training and validation sets
                 valid_fraction: float = 0.1):

        # partition GSC vocabulary into class types
        gsc_vocabulary = set(CLASSES).difference({BACKGROUND_NOISE_LABEL})
        if not set(vocabulary).issubset(gsc_vocabulary):
            raise ValueError(f"The given vocabulary contains words which are not in the GSC vocabulary: {set(vocabulary).difference(gsc_vocabulary)}")
        self._vocabulary = vocabulary
        self._unknown = list(gsc_vocabulary.difference(set(self._vocabulary)))
        self._word_to_index = {w: i for i, w in enumerate([SILENCE_LABEL] + [UNKNOWN_WORD_LABEL] + self._vocabulary)}

        # create and populate data partitions
        self._gsc_partitioner = GSCPartitioner(test_fraction=test_fraction, valid_fraction=valid_fraction)
        self._files_background = []  # background noise waveforms will be used to augment clean speech waveforms, therefore we don't need (and don't want!) to separate them by speaker
        self._files_unknown = {k.value: [] for k in GSCPartition}
        self._files = {k.value: [] for k in GSCPartition}
        self._populate_file_lists(path_data)

        # extend the data sets with silence and unknown word files
        # We assume that the data points representing vocabulary words are
        # represented in such a way that they follow an approximately uniform
        # distribution in each partition set. In this way, we add "silence"
        # and "unknown" data points in such a way that the distribution
        # including also the new classes is still a discrete uniform.
        silence_ratio = 1.0 / len(self._vocabulary)
        unknown_ratio = 1.0 / len(self._vocabulary)
        self._finalise_file_lists(silence_ratio=silence_ratio, unknown_ratio=unknown_ratio)

    def _populate_file_lists(self, path_data: os.PathLike) -> None:

        search_path = os.path.join(path_data, '*', '*.wav')
        for wav_path in glob.glob(search_path):
            gsc_file = get_wav_file_metadata(wav_path)

            if gsc_file.word == BACKGROUND_NOISE_LABEL:
                self._files_background.append(gsc_file)

            else:
                set_ = self._gsc_partitioner.which_set(gsc_file)

                if gsc_file.word in self._unknown:
                    gsc_file = GSCMeta(path=gsc_file.path, word=UNKNOWN_WORD_LABEL, speaker_id=gsc_file.speaker_id)
                    self._files_unknown[set_].append(gsc_file)

                elif gsc_file.word in self._vocabulary:
                    self._files[set_].append(gsc_file)

                else:
                    raise ValueError(f"Word {gsc_file.word} is not a Google Speech Command word. Unexpected folder {os.path.dirname(gsc_file.path)}")

    def _finalise_file_lists(self, silence_ratio: float, unknown_ratio: float):
        """Add *silence* and *unknown* samples to the data sets."""

        random.seed(RANDOM_SEED)

        # we select an arbitrary file since the waveform will anyway be zeroed-out to represent silence
        # gsc_file = self._files[GSCPartition.TEST.value][0]
        # TODO: legacy
        gsc_file = self._files[GSCPartition.TRAIN.value][0]

        for k in GSCPartition:

            set_files = self._files[k.value]
            set_size = len(set_files)

            # add mock-up silence files
            n_silence_files = int(math.ceil(silence_ratio * set_size))
            set_files.extend([GSCMeta(path=gsc_file.path, word=SILENCE_LABEL, speaker_id='') for _ in range(0, n_silence_files)])

            # add randomly chosen unknown word files
            n_unknown_files = int(math.ceil(unknown_ratio * set_size))
            random.shuffle(self._files_unknown[k.value])
            set_files.extend(self._files_unknown[k.value][:n_unknown_files])

    @property
    def word_to_index(self) -> Dict[str, int]:
        return self._word_to_index

    @property
    def files_background(self) -> List[GSCMeta]:
        return self._files_background

    @property
    def files(self) -> Dict[GSCPartition, List[GSCMeta]]:
        return self._files
