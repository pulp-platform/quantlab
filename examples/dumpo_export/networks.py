################################################################################
#   Topology:       CIFAR10
#   Problem:        VGG, ResNet
#   Quantization:   PACT
################################################################################

# Import the get_dataset functions
from systems.CIFAR10.utils.data import load_data_set as load_cifar10
from systems.CIFAR10.utils.transforms import CIFAR10PACTQuantTransform
from systems.CIFAR10.utils.transforms.transforms import CIFAR10STATS

# Import the networks
from systems.CIFAR10.VGG import VGG
from systems.CIFAR10.ViT import ViT
from systems.CIFAR10.ICCT import ICCT
from systems.CIFAR10.ResNet import ResNet as ResNetCIFAR

# Import the quantization functions
from systems.CIFAR10.VGG.quantize import pact_recipe as quantize_vgg, get_pact_controllers as controllers_vgg
from systems.CIFAR10.ViT.quantize import pact_recipe as quantize_vit, get_pact_controllers as controllers_vit
from systems.CIFAR10.ICCT.quantize import pact_recipe as quantize_icct, get_pact_controllers as controllers_icct
from systems.CIFAR10.ResNet.quantize import pact_recipe as quantize_resnet_cifar, get_pact_controllers as controllers_resnet_cifar

################################################################################
#   Topology:       MNIST
#   Problem:        simpleCNN
#   Quantization:   PACT
################################################################################

# Import the get_dataset functions
from systems.MNIST.utils.data import load_data_set as load_mnist
from systems.MNIST.utils.transforms import MNISTPACTQuantTransform
from systems.MNIST.utils.transforms.transforms import MNISTSTATS

# Import the networks
from systems.MNIST.simpleCNN import simpleCNN

# Import the quantization functions
from systems.MNIST.simpleCNN.quantize import pact_recipe as quantize_simpleCNN, get_pact_controllers as controllers_simpleCNN

################################################################################
#   Topology:       ILSVRC12
#   Problem:        MobileNetV1, MobileNetV2, MobileNetV3, ResNet
#   Quantization:   PACT
################################################################################

# Import the get_dataset functions
from systems.ILSVRC12.utils.data import load_ilsvrc12
from systems.ILSVRC12.utils.transforms import ILSVRC12PACTQuantTransform
from systems.ILSVRC12.utils.transforms.transforms import ILSVRC12STATS

# Import the networks
from systems.ILSVRC12.MobileNetV1 import MobileNetV1
from systems.ILSVRC12.MobileNetV2 import MobileNetV2
from systems.ILSVRC12.MobileNetV3 import MobileNetV3
from systems.ILSVRC12.ResNet import ResNet

# Import the quantization functions
from systems.ILSVRC12.MobileNetV1.quantize import pact_recipe as quantize_mnv1, get_pact_controllers as controllers_mnv1
from systems.ILSVRC12.MobileNetV2.quantize import pact_recipe as quantize_mnv2, get_pact_controllers as controllers_mnv2
from systems.ILSVRC12.MobileNetV3.quantize import pact_recipe as quantize_mnv3, get_pact_controllers as controllers_mnv3
from systems.ILSVRC12.ResNet.quantize import pact_recipe as quantize_resnet, get_pact_controllers as controllers_resnet

################################################################################
#   Topology:       DVS128
#   Problem:        DVSHybridNet
#   Quantization:   PACT
################################################################################

# Import the get_dataset functions
from systems.DVS128.dvs_cnn.preprocess import load_data_set as load_dvs128, DVSAugmentTransform

# Import the networks
from systems.DVS128.dvs_cnn import DVSHybridNet, get_input_shape as get_in_shape_dvsnet

# Import the quantization functions
from systems.DVS128.dvs_cnn.quantize import pact_recipe as quantize_dvsnet, get_pact_controllers as controllers_dvsnet

