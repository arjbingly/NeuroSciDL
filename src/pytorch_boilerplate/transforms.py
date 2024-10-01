from typing import Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor.

    Args:
        mean (Optional[float], optional): Mean of the Gaussian noise. If None, the mean of the tensor is used. Defaults to None.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.5.
    """
    def __init__(self, mean: Optional[float] = None, std: float = 0.5):
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

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns:
            str: String representation of the object.
        """
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CategoricalEncoder(object):
    def __init__(self, categorical_cols, seperate_nominal_df: bool = True, sparse_nominal: bool = True):
        self.categorical_cols = categorical_cols
        self.df = None
        self.original_df = None
        self.ordinal_cols = None
        self.nominal_cols = None
        self.seperate_nominal_df = seperate_nominal_df
        self.sparse_nominal = sparse_nominal

    def _make_cols_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Checks if the provided categorical columns are categorical, if not, makes them categorical.  # noqa: D205  # noqa: D205
        Note that: this assumes that the columns are nominal categorical variables.
        """
        for col in self.categorical_cols:
            if not df[col].dtype == 'category':
                df[col] = df[col].astype('category')
        return df

    def _find_categorical_type(self, df: pd.DataFrame) -> None:
        """Finds the ordinal and nominal categorical varibles/columns in the dataframe."""
        self.ordinal_cols = [col for col in self.categorical_cols if df[col].ordered]
        self.nominal_cols = [col for col in self.categorical_cols if not df[col].ordered]

    def _encode_nominal(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, columns=self.nominal_cols, drop_first=True, sparse=self.sparse_nominal)

    def _encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        df_dict = {}
        for col in self.ordinal_cols:
            df_dict[col] = df[col].cat.codes  # Label encoding
        return pd.DataFrame(df_dict)

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        return self.fit_transform(df)

    def fit(self, df: pd.DataFrame) -> None:
        df = self._make_cols_categorical(df)
        self._find_categorical_type(df)

    def transform(self, df: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        nominal_df = self._encode_nominal(df)
        ordinal_df = self._encode_ordinal(df)
        if self.seperate_nominal_df:
            return nominal_df, ordinal_df
        else:
            df = pd.concat([nominal_df, ordinal_df], axis=1)
            df = df.drop(columns=self.categorical_cols)
            return pd.concat([nominal_df, ordinal_df], axis=1)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def __repr__(self):
        return self.__class__.__name__ + f'(categorical_columns={self.categorical_cols}, seperate_nominal_df={self.seperate_nominal_df})'

def resnet_preprocess(img: Image) -> Tensor:
    img = img.convert('L')
    img = Image.merge("RGB", (img, img, img))
    _preprocess = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), AddGaussianNoise(std=0.5),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    img = _preprocess(img)
    return img
