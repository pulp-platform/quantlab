import argparse
from pathlib import Path
import torch
import torchvision as tv
from mobilenetv3 import MobileNetV3

def convert_key(k):
    if k.startswith('features.0'):
        k = k.replace('features.0', 'pilot')
    elif k.startswith('features'):
        ksplit = k.split('.')
        n = int(ksplit[1])
        k = f'features.{n-1}.{".".join(ksplit[2:])}'
    k = k.replace('fc1', 'lin.0')
    k = k.replace('fc2', 'lin.2')
    return k



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV3 checkpoint conversion script")
    parser.add_argument('--config', '-c',
                        type=str,
                        default='small',
                        help='MobileNetV3 configuration - can be "small" or "large"')
    parser.add_argument('--ckpt_in', '-i',
                        type=str,
                        default=None,
                        help='Checkpoint of a full-precision, torchvision-style MobileNetV3. If left empty, the pretrained model with width multiplier from the torchvision repository will be used.')
    parser.add_argument('--ckpt_out', '-o',
                        type=str,
                        default='./pretrained/MNv3_small_224_1.0.ckpt',
                        help='Output checkpoint for QuantLab\'s MobileNetV3 implementation.')

    args = parser.parse_args()

    if not args.ckpt_in:
        if args.config == 'small':
            tv_mnv3 = tv.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
        else:
            tv_mnv3 = tv.models.mobilenetv3.mobilenet_v3_large(pretrained=True)
        state_dict_in = tv_mnv3.state_dict()
    else:
        state_dict_in = torch.load(args.ckpt_in)

    Path(args.ckpt_out).parent.mkdir(exist_ok=True, parents=True)

    my_mnv3 = MobileNetV3(args.config)
    new_state_dict = {convert_key(k): v for k, v in state_dict_in.items()}

    bad_key = False
    for k1, k2 in zip(my_mnv3.state_dict().keys(), new_state_dict.keys()):
        if k1 != k2:
            print(f"QuantLab MNV3 key: {k1}, converted MNV3 key: {k2}")
            bad_key = True

    if not bad_key:
        for k, v in new_state_dict.items():
            new_state_dict[k] = v.reshape(my_mnv3.state_dict()[k].shape)
        print(f"Conversion successful; Saving checkpoint at {args.ckpt_out}")
        torch.save(new_state_dict, args.ckpt_out)
