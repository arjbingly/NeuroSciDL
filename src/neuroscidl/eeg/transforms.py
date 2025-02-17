import json
from os import PathLike

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List
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

def read_transform_config(config_path: Union[str, PathLike], dataset_key: str, query_keys: Union[List[str], str]):
        result = []
        with open(config_path, 'r') as f:
            noise_config = json.load(f)
        if isinstance(query_keys, str):
            query_keys = [query_keys]
        if noise_config.get(dataset_key) is None:
            raise ValueError(f'No config found for dataset {dataset_key}')
        else:
            for key in query_keys:
                _result = noise_config[dataset_key].get(key)
                if _result is None:
                    raise ValueError(f'No config found for key {key} in dataset {dataset_key}')
                else:
                    result.append(_result)
        return result

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
                 config_path: Optional[Union[str, PathLike]] = None,
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
            self.mean, self.std = read_transform_config(config_path = self.config_path,
                                                        dataset_key=self.config_key,
                                                        query_keys=['mean', 'std'])
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

class ZTransform(object):
    def __init__(self, config_path: Optional[Union[str, PathLike]] = None,
                 config_key: Optional[str]=None,
                 mean: float = 0.0,
                 std: float = 1.0,
                 with_std=True,
                 with_mean=True):
        # TODO: resolve mean and std from config given with_std and with_mean
        self.config_path = config_path
        self.config_key = config_key
        self.mean = mean
        self.std = std
        self.with_std = with_std
        self.with_mean = with_mean
        if self.config_path is not None:
            if self.config_key is None:
                raise ValueError('config_key must be provided if config_path is provided')
            self.mean, self.std = read_transform_config(config_path = self.config_path,
                                                        dataset_key=self.config_key,
                                                        query_keys=['mean', 'std'])
        else:
            self.mean = mean
            self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        return (tensor - self.mean) / self.std
