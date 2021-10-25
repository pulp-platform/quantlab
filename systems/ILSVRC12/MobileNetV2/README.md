# MobileNetV2 for ImageNet
This topology package implements quantized [MobileNetV2](https://arxiv.org/abs/1801.04381) training on the [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/) dataset. The quantization algorithms supported are [PACT/SAWB](https://arxiv.org/pdf/1807.06964) and [TQT](https://arxiv.org/abs/1903.08066).
## Quantization
The MobileNetV2 quantization flow can be considered the reference `QuantLab` flow for PACT/TQT. Under the
`network`->`quantize` in the `config.json`, the configuration for each quantized layer type can be specified for each layer individually by the respective `torch.Module`'s hierarchical name. Under the `PACTConv2d` key, you may enter the parameters passed to the constructor of that class for each layer individually. The activation layers (`PACTUnsignedAct`) and the linear layer (`PACTLinear`) are treated equivalently. For each class, the default arguments (used if the corresponding keys are not present in the layer's configuration) can be entered under the `kwargs` key.

The quantization routine `pact_recipe` in [quantize/pact.py] works by first replacing each layer with its fake-quantized counterpart according to the supplied configuration, then running a harmonization pass on the network to ensure that the resulting network can be integerized correctly.
## Configurations
3 example configurations are provided:
* No quantization - `config.fp32.json`. Note that this configuration may not reach the published accuracy; we conducted our experiments starting from a pre-trained `torchvision` model.
* Quantization with PACT - `config.pact.json`
* Quantization with TQT - `config.tqt.json`
### Pretrained full-precision network
Under the `network`->`pretrained` key, you can specify a full-precision checkpoint to initialize the network. As our MobileNetV2 module does not have the same naming convention as the `torchvision` implementation, we provide a small script ([mnv2_convert_checkpoint.py]) to generate a valid `QuantLab` MNv2 checkpoint from a pretrained `torchvision` network.

### Accuracy results
All results are for MobileNetV2 with input resolution 224x224 and width modifier 1.0.

| Quant. Algo. | Precision | Accuracy |
| ------------ | --------- | -------- |
| None         | FP32      |    72.0% |
| PACT         | 8b/8b     |    71.4% |
| TQT          | 8b/8b     |    71.4% |

