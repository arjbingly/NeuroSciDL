from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


class AddGaussianNoise(object):
    def __init__(self, mean: Optional[float] = None, std: float = 0.5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.mean is None:
            mean = tensor.mean()
            return tensor + torch.randn(tensor.size()) * self.std + mean
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def preprocess(img: Image) -> Tensor:
    img = img.convert('L')
    img = Image.merge("RGB", (img, img, img))
    _preprocess = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), AddGaussianNoise(std=0.5),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    img = _preprocess(img)
    return img
