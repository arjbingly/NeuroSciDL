import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
from torch import Tensor

##
# Based on data used:
# Mean: 0.008104733313294322
# Std: 0.025660938145368836

experimental_mean = 0.008104733313294322
experimental_std = 0.025660938145368836

default_mean = round(experimental_mean, 6)
default_std = round(experimental_std, 6) * 0.5

class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor.

    Args:
        mean (Optional[float], optional): Mean of the Gaussian noise. If None, the mean of the tensor is used. Defaults to None.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.5.
    """
    def __init__(self, mean: Optional[float] = default_mean, std: float = default_std):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: Tensor) -> Tensor:
        """Applies Gaussian noise to the tensor.

        Args:
            tensor (Tensor): Input tensor to which noise will be added.

        Returns:
            Tensor: Tensor with added Gaussian noise.
        """
        if self.mean is None:
            mean = tensor.mean()
            return tensor + torch.randn(tensor.size()) * self.std + mean
        else:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
