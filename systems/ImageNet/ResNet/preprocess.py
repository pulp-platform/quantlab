import os
import torch
import torchvision
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, Compose


_ImageNet = {
    'Normalize': {
        'mean': (0.485, 0.456, 0.406),
        'std':  (0.229, 0.224, 0.225)
    },
    'PCA': {
        'eigvals': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvecs': torch.Tensor([[-0.5675,  0.7192,  0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948,  0.4203]])
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
        gs[0].mul_(self._Rec601['red']).add_(self._Rec601['green'], gs[1]).add_(self._Rec601['blue'], gs[2])
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
        # when alpha = 0., the image does not change
        # when alpha = alphamax (<= 1.), the image is replaced by the average of pixels of its grayscale version
        gs    = self.grayscale(img)
        gs.fill_(gs.mean())
        alpha = self.alphamax * torch.rand(1).item()
        return torch.lerp(img, gs, alpha)


class Saturation(object):

    def __init__(self, alphamax):
        self.alphamax  = alphamax
        self.grayscale = Grayscale()

    def __call__(self, img):
        # when alpha = 0., the image does not change
        # when alpha = alphamax (<= 1.), the image is replaced by its grayscale version
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

    def __init__(self, pcaparams, alphastd=0.1):
        self.eigvals  = pcaparams['eigvals']
        self.eigvecs  = pcaparams['eigvecs']
        self.alphastd = alphastd

    def __call__(self, img):
        # let V be the matrix which columns V^{(j)} are the Principal Components
        # to each RGB pixel is added a random combination \sum_{j} V^{(j)} (\alpha_{j} * \Lambda_{j}),
        # with \alpha_{j} a normally distributed random scaling factor of the j-th component
        if self.alphastd != 0.:
            alpha = img.new_tensor(0).resize_(3).normal_(0, self.alphastd)
            noise = torch.mul(alpha.view(1, 3), self.eigvals.view(1, 3))
            noise = torch.mul(self.eigvecs.type_as(img).clone(), noise).sum(1)
            img   = torch.add(img, noise.view(3, 1, 1).expand_as(img))
        return img


def get_transforms(augment):
    train_t = Compose([RandomResizedCrop(224),
                       RandomHorizontalFlip(),
                       ToTensor(),
                       ColorJitter(),
                       Lighting(_ImageNet['PCA']),
                       Normalize(**_ImageNet['Normalize'])])
    valid_t = Compose([Resize(256),
                       CenterCrop(224),
                       ToTensor(),
                       Normalize(**_ImageNet['Normalize'])])
    if not augment:
        train_t = valid_t
    transforms = {
        'training':   train_t,
        'validation': valid_t
    }
    return transforms


def load_data_sets(logbook):
    transforms = get_transforms(logbook.config['data']['augment'])
    train_set  = torchvision.datasets.ImageFolder(os.path.join(logbook.dir_data, 'train'), transforms['training'])
    valid_set  = torchvision.datasets.ImageFolder(os.path.join(os.path.realpath(logbook.dir_data), 'val'), transforms['validation'])
    return train_set, valid_set
