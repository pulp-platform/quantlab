# 
# transforms.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich. All rights reserved.
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

from torchvision.transforms import Normalize


# These values are the per-channel mean and standard deviation of SVHN images
# after their values have been mapped from the `UINT8` integer range ({0, ...,
# 255}) to the floating-point range [0, 1].
SVHNCHANNELSTATS =\
    {
        'normalize':
            {
                'mean': (0.4377, 0.4438, 0.4728),
                'std':  (0.1980, 0.2010, 0.1970)
            }
    }


# These values are the mean and standard deviation of all the pixels in SVHN
# images after their values have been mapped from the `UINT8` integer range
# ({0, ..., 255}) to the floating-point range [0, 1].
SVHNIMAGESTATS =\
    {
        'normalize':
            {
                'mean': (0.4514,),
                'std':  (0.1993,)
            }
    }


class SVHNNormalizeChannels(Normalize):
    def __init__(self):
        super(SVHNNormalizeChannels, self).__init__(**SVHNCHANNELSTATS['normalize'])


class SVHNNormalizeImage(Normalize):
    def __init__(self):
        super(SVHNNormalizeImage, self).__init__(**SVHNIMAGESTATS['normalize'])
