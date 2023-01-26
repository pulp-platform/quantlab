# 
# export_net.py
# 
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# 
# Copyright (c) 2023 ETH Zurich.
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

import torch
import argparse
import onnx
import os
import numpy as np

# Set the PYTHONPATH to include QuantLab's root directory
import sys
from pathlib import Path
_QL_ROOTPATH = Path(__file__).absolute().parent.parent.parent.parent

sys.path.append(str(_QL_ROOTPATH))

# Import the get_dataset functions for CIFAR10 
from systems.CIFAR10.utils.data import load_data_set as load_cifar10

from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical augmentation transforms
from torchvision.transforms import ToTensor              # structural transforms
from torchvision.transforms import Resize              # structural transforms

from systems.CIFAR10.utils.transforms import CIFAR10NormalizeHomogeneous  # public CIFAR-10 statistics

# Import the networks
from systems.CIFAR10.CCT import CCT

import onnxruntime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.transformers.optimizer import optimize_model
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    enable_gelu: bool = True
    enable_layer_norm: bool = True
    enable_attention: bool = True
    enable_skip_layer_norm: bool = True
    enable_embed_layer_norm: bool = True
    enable_bias_skip_layer_norm: bool = True
    enable_bias_gelu: bool = True
    enable_gelu_approximation: bool = False
    enable_qordered_matmul: bool = True
    enable_shape_inference: bool = True
    attention_mask_format: int = 3

class TransformB(Compose):

    def __init__(self, augment: bool, image_size=32 ):

        transforms = []
        if augment:
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(CIFAR10NormalizeHomogeneous())
        transforms.append(Resize(image_size))

        super(TransformB, self).__init__(transforms)

def export_onnx_cct(args):
    model = CCT()

    print(model)

    model = model.eval()
    
    out_path = Path(os.path.join(str(Path(__file__).parent), args.export_dir))
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = f"{args.export_name}.onnx"
    onnx_path = out_path.joinpath(onnx_file)

    print("Exporting to", onnx_path)

    path_data = _QL_ROOTPATH.joinpath('systems').joinpath("CIFAR10").joinpath('data')

    transform_inst = TransformB(augment=False, image_size=32)
    ds = load_cifar10(partition='valid', path_data=str(path_data), n_folds=1, current_fold_id=0, cv_seed=0, transform=transform_inst)

    in_data: torch.Tensor = ds[42][0].unsqueeze(0)

    # First export an ONNX graph without shape inference
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
    }
    try:
        torch.onnx._export(model.to('cpu'), (in_data, ),
                           onnx_path,
                           opset_version=12,
                           onnx_shape_inference=True,
                           verbose=False,
                           **kwargs)

    except torch.onnx.CheckerError:
        print("Disregarding PyTorch ONNX CheckerError...")

    optimization_config = OptimizationConfig(
        enable_skip_layer_norm=False, 
        enable_bias_gelu=False,
    )
    optimizer =  optimize_model(str(onnx_path), optimization_options=optimization_config, opt_level=1)
    optimizer.optimize
    optimizer.save_model_to_file(str(onnx_path))

    onnxModel = onnx.load_model(onnx_path)
    onnxModel = SymbolicShapeInference.infer_shapes(onnxModel)
    onnx.save_model(onnxModel, onnx_path)

    session = onnxruntime.InferenceSession(str(onnx_path))

    # Check if both model have the same output
    result1 = session.run(["output"], {"input": in_data.numpy()})
    result2 = model.forward(in_data).detach().numpy()
    print("Results are the same:", np.allclose(result1,result2, atol=1e-6))
    print(" => Abs Diff:", np.max(np.abs(result2-result1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT ONNX Export')
    parser.add_argument('--config', metavar='config', type=str, default="Tiny", choices=["Small", "Base", "Tiny"],
                    help='Configuration String ("Small", "Base")')
    parser.add_argument('--export_dir', type=str, default="./",
                        help='Export the integerized network to the specified directory.')
    parser.add_argument('--export_name', type=str, default="network",
                        help='Name of the exported ONNX graph. By default, this is identical to the value of the "--net" flag')

    args = parser.parse_args()

    export_onnx_cct(args)
