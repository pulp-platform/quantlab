import numpy as np
from torchvision.transforms import Compose, RandomAffine
from itertools import product

class AddTernaryNoise(object):
    # add ternary noise to a ternary tensor - a non-ternary tensor will stay non-ternary!
    def __call__(self, sample):
        n = np.random.choice(np.array([-1,0,1], dtype=sample.dtype), sample.shape, p=[0.01, 0.98, 0.01])
        sample += n
        sample = np.clip(sample, -1, 1)
        return sample

class TranslateTransform(object):
    def __init__(self, fracs):
        self.fracs = fracs

    # translate numpy array by relative fraction of height & width
    def __call__(self, sample):
        max_dx = int(sample.shape[2]*self.fracs[1])
        max_dy = int(sample.shape[1]*self.fracs[0])
        oy = sample.shape[1]
        ox = sample.shape[2]
        dx = np.random.randint(-max_dx, max_dx+1)
        dy = np.random.randint(-max_dy, max_dy+1)
        sample = np.pad(sample, ((0,0), (np.max([dy, 0]), np.max([-dy, 0])), (np.max([dx, 0]), np.max([-dx, 0]))))
        sample = sample[:, np.max([-dy, 0]):np.max([oy-dy, oy]), np.max([-dx, 0]):np.max([ox-dx, ox])]
        return sample

class RandomPxFlipTransform(object):
    #flip every pixel with probability p IID
    def __init__(self, p : float):
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        nz_idxs = sample!=0
        facs = 2*self.rng.binomial(1, p=1-self.p, size=np.sum(nz_idxs))-1
        sample[nz_idxs] *= facs
        return sample


class TernaryDownsampleTransform(object):
    def __init__(self, low, high, factor):
        self.low = low
        self.high = high
        self.factor = factor

    def __call__(self, sample):
        if self.factor == 1:
            return sample
        # assume 3 dimensional array
        # shape: (k, x, y)
        # first step: get all the subsampled arrays, starting from different pixels
        subsampled = np.stack([sample[:, start_y::self.factor, start_x::self.factor] for start_y, start_x in product(range(self.factor), range(self.factor))], axis=0)
        # second step: sum the subsampled arrays together
        # shape: (k, x/factor, y/factor)
        summed = subsampled.sum(axis=0)
        out = np.zeros_like(summed, dtype=np.int)
        # third step: threshold the summed array to get ternary result
        out[summed<self.low] = -1
        out[summed>self.high] = 1
        return out



class DVSAugmentTransform(Compose):
    def __init__(self, augment : bool, downsample : int = 1, p_flip : float = 0.):
        transforms = []
        if augment:
            transforms.append(RandomPxFlipTransform(p=p_flip))
            transforms.append(TranslateTransform(fracs=(0.1, 0.1)))
        n_vals_in_range = (2*downsample**2+1)//3
        transforms.append(TernaryDownsampleTransform(-downsample**2+n_vals_in_range, downsample**2-n_vals_in_range, downsample))
        super(DVSAugmentTransform, self).__init__(transforms)
