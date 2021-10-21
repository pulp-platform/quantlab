#
# mnv2_convert_checkpoint.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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

import argparse
from pathlib import Path

import torch
import torchvision as tv

from mobilenetv2 import MobileNetV2

def convert_key(k):
    if k.startswith('features.0'):
        k = k.replace('features.0', 'pilot')
    elif k.startswith('features'):
        ksplit = k.split('.')
        n = int(ksplit[1])
        k = f"features.{n-1}.{'.'.join(ksplit[2:])}"

    k = k.replace('conv', 'residual_branch')
    if k.startswith('features.0'):
        k = k.replace('residual_branch.1', 'residual_branch.1.0')
        k = k.replace('residual_branch.2', 'residual_branch.1.1')
    else:
        k = k.replace('residual_branch.2', 'residual_branch.2.0')
        k = k.replace('residual_branch.3', 'residual_branch.2.1')
    return k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV2 checkpoint conversion script")
    parser.add_argument('--ckpt_in', '-i',
                        type=str,
                        default=None,
                        help='Checkpoint of a full-precision, torchvision-style MobileNetV2. If left empty, the pretrained model with width multiplier 1.0 from the torchvision repository will be used.')
    parser.add_argument('--ckpt_out', '-o',
                        type=str,
                        default='./pretrained/MNv2_224_1.0.ckpt',
                        help='Output checkpoint for QuantLab\'s MobileNetV2 implementation.')

    args = parser.parse_args()

    if not args.ckpt_in:
        tv_mnv2 = tv.models.mobilenet_v2(pretrained=True)
        state_dict_in = tv_mnv2.state_dict()

    Path(args.ckpt_out).parent.mkdir(exist_ok=True, parents=True)

    my_mnv2 = MobileNetV2('standard', 1.0)
    new_state_dict = {convert_key(k): v for k, v in state_dict_in.items()}

    for k1, k2 in zip(my_mnv2.state_dict().keys(), new_state_dict.keys()):
        if k1 != k2:
            print(f"QuantLab MNV2 key: {k1}, converted MNV2 key: {k2}")


    torch.save(new_state_dict, args.ckpt_out)
