# 
# transforms.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich.
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
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip          # statistical augmentation transforms
from torchvision.transforms import Resize, CenterCrop, ToTensor  # structural transforms
from torchvision.transforms import RandomResizedCrop             # evil transforms: they combine statistical augmentation with structural aspects


ILSVRC12STATS = \
    {
        'normalize':
            {
                'mean': (0.485, 0.456, 0.406),
                'std':  (0.229, 0.224, 0.225)
            },
        'PCA':
            {
                'eigvals': torch.Tensor([0.2175, 0.0188, 0.0045]),
                'eigvecs': torch.Tensor([[-0.5675,  0.7192,  0.4009],
                                         [-0.5808, -0.0045, -0.8140],
                                         [-0.5836, -0.6948,  0.4203]])
            },
        'quantize':
            {
                'min': -2.1179039478,
                'max': 2.6400001049,
                'eps': 0.020625000819563866
            }
    }


class Grayscale(object):

    def __init__(self):
        self._Rec601 = {
            'red':   0.299,
            'green': 0.587,
            'blue':  0.114
        }

    def __call__(self, img):
        # uses the Recommendation 601 (Rec. 601) RGB-to-YCbCr conversion
        gs = img.clone()
        gs[0].mul_(self._Rec601['red']).add_(gs[1], alpha=self._Rec601['green']).add_(gs[2], alpha=self._Rec601['blue'])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Brightness(object):

    def __init__(self, alphamax):
        self.alphamax = alphamax

    def __call__(self, img):
        # when alpha = 0., the image does not change
        # when alpha = alphamax (<= 1.), the image goes black
        gs    = torch.zeros_like(img)
        alpha = self.alphamax * torch.rand(1).item()
        return torch.lerp(img, gs, alpha)


class Contrast(object):

    def __init__(self, alphamax):
        self.alphamax  = alphamax
        self.grayscale = Grayscale()

    def __call__(self, img):
        # when alpha = 0. the image does not change
        # when alpha = alphamax (<= 1.) the image is replaced by the average of pixels of its grayscale version
        gs    = self.grayscale(img)
        gs.fill_(gs.mean())
        alpha = self.alphamax * torch.rand(1).item()
        return torch.lerp(img, gs, alpha)


class Saturation(object):

    def __init__(self, alphamax):
        self.alphamax  = alphamax
        self.grayscale = Grayscale()

    def __call__(self, img):
        # when alpha = 0. the image does not change
        # when alpha = alphamax (<= 1.) the image is replaced by its grayscale version
        gs    = self.grayscale(img)
        alpha = self.alphamax * torch.rand(1).item()
        return torch.lerp(img, gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness_amax=0.4, contrast_amax=0.4, saturation_amax=0.4):
        self.transforms = []
        if brightness_amax != 0.:
            self.transforms.append(Brightness(alphamax=brightness_amax))
        if contrast_amax != 0.:
            self.transforms.append(Contrast(alphamax=contrast_amax))
        if saturation_amax != 0.:
            self.transforms.append(Saturation(alphamax=saturation_amax))

    def __call__(self, img):
        if self.transforms is not None:
            order = torch.randperm(len(self.transforms))
            for i in order:
                img = self.transforms[i](img)
        return img


class Lighting(object):
    """AlexNet-style, PCA-based lighting noise."""

    def __init__(self, eigvals, eigvecs, alphastd=0.1):
        self.__eigvals  = eigvals
        self.__eigvecs  = eigvecs
        self.__alphastd = alphastd

    def __call__(self, img):
        # Let V be the matrix whose columns V^{(j)} are the principal
        # components of the 3D point cloud consisting of all the RGB-encoded
        # pixels in the ILSVRC12 data set images (rescaled from {0, ..., 255}
        # to [0, 1]) and \Lambda be the (diagonal) matrix of eigenvalues.
        # This transform takes an image and adds a random linear combination
        # \sum_{j=1}^{3} \alpha_{j} * \Lambda_{jj} * V^{(j)} to each
        # RGB-encoded pixel, where \alpha_{j} a normally distributed random
        # scaling factor of the j-th component.
        if self.__alphastd != 0.:
            alpha = img.new_tensor(0).resize_(3).normal_(0, self.__alphastd)
            noise = torch.mul(alpha.view(1, 3), self.__eigvals.view(1, 3))
            noise = torch.mul(self.__eigvecs.type_as(img).clone(), noise).sum(1)
            img   = torch.add(img, noise.view(3, 1, 1).expand_as(img))
        return img


class ILSVRC12Lighting(Lighting):
    def __init__(self):
        super(ILSVRC12Lighting, self).__init__(**ILSVRC12STATS['PCA'])


class ILSVRC12Normalize(Normalize):
    def __init__(self):
        super(ILSVRC12Normalize, self).__init__(**ILSVRC12STATS['normalize'])


class ILSVRC12AugmentTransform(Compose):

    def __init__(self,
                 augment:    bool,
                 image_size: int = 224):

        # validate arguments
        _NON_AUGMENT_RESIZE = 256
        if image_size > _NON_AUGMENT_RESIZE:
            raise ValueError  # otherwise, `CenterCrop` won't work in the non-augmented branch

        if augment:
            transforms = [RandomResizedCrop(image_size),
                          RandomHorizontalFlip(),
                          ToTensor(),
                          ColorJitter(),
                          ILSVRC12Lighting(),
                          ILSVRC12Normalize()]

        else:
            transforms = [Resize(_NON_AUGMENT_RESIZE),
                          CenterCrop(image_size),
                          ToTensor(),
                          ILSVRC12Normalize()]

        super(ILSVRC12AugmentTransform, self).__init__(transforms)
