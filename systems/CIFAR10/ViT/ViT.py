# 
# simplecnn.py
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
import torch.nn as nn

# Import custom version of ViT Model
from .model.modeling_vit import ViTForImageClassification, ViTConfig

import quantlib.editing.lightweight as qlw


_CONFIGS = {
    'Tiny': {
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 256,
        "hidden_act": "gelu",
        "image_size": 28,
        "patch_size": 4,
        "qkv_bias": True,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    },
    'Small': {
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "hidden_act": "gelu",
        "image_size": 28,
        "patch_size": 4,
        "qkv_bias": True,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    },
    'Base': {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "image_size": 32,
        "patch_size": 4,
        "qkv_bias": True,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1
    },
}

class ViT(nn.Module):
    def __init__(self, config: str, num_classes: int = 10, pretrained : str = None, seed: int = -1) -> None:
        super(ViT, self).__init__(    )
       
        if config in _CONFIGS:
            self.config: ViTConfig = ViTConfig(**_CONFIGS[config])
        else:
            raise KeyError(f"Config '{config}' not found!")  

        self.config.num_labels = num_classes 

        self.model: ViTForImageClassification = ViTForImageClassification(self.config)

        if pretrained is not None:
            state_dict = torch.load(pretrained)
            if 'net' in state_dict:
                self.load_state_dict(state_dict['net'])
            else:
                self.load_state_dict(state_dict)
        else:
            self._initialize_weights(seed)

        lwg = qlw.LightweightGraph(self.model)
        
        print("=== Traced Network ===")
        lwg.show_nodes_list()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward(pixel_values=x)
        return outputs.logits

    def _initialize_weights(self, seed : int = -1):

        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    
