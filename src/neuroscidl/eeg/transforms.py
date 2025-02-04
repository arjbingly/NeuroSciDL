import json

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import torch
from torch import Tensor
from torchvision.transforms.v2 import GaussianNoise

##
# Based on data used:
# Mean: 0.008104733313294322
# Std: 0.025660938145368836

experimental_mean = 0.008104733313294322
experimental_std = 0.025660938145368836

default_mean = round(experimental_mean, 6)
default_std = round(experimental_std, 6) * 0.5

class GaussianNoiseTransform(object):
    """Transform to add Gaussian noise to a tensor.
    This transform can be initialized with a configuration file and key to get the mean and std of the noise.
    Args:
        config_path (Optional[Union[str, Path]]): Path to the configuration file.
        config_key (Optional[str]): Key to the configuration in the file.
        mean (Optional[float], optional): Mean of the Gaussian noise.
        std (float, optional): Standard deviation of the Gaussian noise.
    """
    def __init__(self,
                 config_path: Optional[Union[str, Path]] = None,
                 config_key: Optional[str]=None,
                 mean: Optional[float] = default_mean,
                 std: float = default_std,
                 sig_digits:int = 6,
                 clip: bool = False):
        self.config_path = config_path
        self.config_key = config_key
        self.sig_digits = sig_digits
        self.clip = clip
        if self.config_path is not None:
            if self.config_key is None:
                raise ValueError('config_key must be provided if config_path is provided')
            with open(config_path, 'r') as f:
                noise_config = json.load(f)
            self.mean = noise_config[config_key]['mean']
            self.std = noise_config[config_key]['std']
        else:
            self.mean = mean
            self.std = std
        self.mean = round(self.mean, self.sig_digits)
        self.std = round(self.std/2, self.sig_digits)

        self.noise = GaussianNoise(mean, std, clip)

    def __call__(self, tensor: Tensor) -> Tensor:
        """Applies Gaussian noise to the tensor.

        Args:
            tensor (Tensor): Input tensor to which noise will be added.

        Returns:
            Tensor: Tensor with added Gaussian noise.
        """
        return self.noise(tensor)

    def __repr__(self) -> str:
        if self.config_path is not None:
            return (self.__class__.__name__ +
                    f'(config_path={self.config_path}, config_key={self.config_key}, mean={self.mean}, std={self.std})')
        else:
            return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
