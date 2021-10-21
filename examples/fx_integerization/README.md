# FX Integerization Example Script
The `integerize_pactnets.py` script allows you to load checkpoints, quantize, evaluate, integerize and export integerized ONNX graphs of the following topologies/problems:
* MobileNetV1/ImageNet
* MobileNetV2/ImageNet
* VGG9/CIFAR-10

You can specify the network (the problem is implicit as each network only targets 1 problem), QuantLab experiment ID, export directory and name and whether to validate the chosen network before and/or after integerization. For example, to integerize checkpoint `42` of experiment `13` of the ILSVRC12/MobileNetV1 topology, validating the accuracy after integerization (but not before), and exporting it to the directory `my_mnv1` run the following command:
```bash
python integerize_pactnets.py --net MobileNetV1 --exp_id 13 --ckpt_id 42 --validate_tq --export_dir my_mnv1
```
For additional information on the command line flags, run
```bash
python integerize_pactnets.py --help
```

This script should run without issues in the Conda environment specified by [quantlab.yml](../../quantlab.yml).
