from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from transforms import CategoricalEncoder


class ImageDataset(Dataset):
    """Custom Dataset for loading images and their annotations.

    Args:
        image_dir (Union[str, Path]): Directory where images are stored.
        annotations_df (pd.DataFrame): DataFrame containing image filenames and their corresponding labels.
        transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        target_transform (callable, optional): A function/transform to apply to the labels. Defaults to None.
    """

    def __init__(self, image_dir: Union[str, Path], annotations_df: pd.DataFrame, transform=None, target_transform=None
                 # TODO: type hint for transform
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.annotations_df = annotations_df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.annotations_df)

    def __getitem__(self, idx: int) -> Tuple[Image, int]:
        """Fetches the image and label at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[Image, int]: A tuple containing the image and its label.
        """
        img_path = self.image_dir / self.annotations_df.iloc[idx, 0]
        label = self.annotations_df.iloc[idx, 1]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class ImageDataModule(L.LightningDataModule):
    """LightningDataModule for loading and processing image data.

    Args:
        image_dir (Union[str, Path]): Directory where images are stored.
        annotation_file (Union[str, Path]): Path to the CSV file containing image annotations.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        filename_col (Optional[str], optional): Column name for filenames in the annotation file. Defaults to None.
        label_col (Optional[str], optional): Column name for labels in the annotation file. Defaults to None.
        transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        target_transform (callable, optional): A function/transform to apply to the labels. Defaults to None.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Notes:
        - If 'filename_col' and or 'label_col' is not provided, the first and second columns of the annotation file are used.
        - The 'filename_col' column should contain the paths to the image files relative to the 'image_dir'.
        - The default transform converts images to tensors.

    Todo:
        - Add support for custom transforms
        - Add support for test and predict datasets
    """

    def __init__(self,
                 image_dir: Union[str, Path],
                 annotation_file: Union[str, Path],
                 batch_size: int = 32,
                 filename_col: Optional[str] = None,
                 label_col: Optional[str] = None,
                 transform=None,
                 # TODO: type hint for transform
                 target_transform=None,  # TODO: type hint for transform
                 num_workers: int = 0):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.annotations_file = annotation_file
        self.batch_size = batch_size
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.target_transform = target_transform

        self._get_cols(filename_col, label_col)

        self.num_workers = num_workers
        self.input_size = None
        self.output_size = None

    def _get_cols(self, filename_col, label_col) -> None:
        """Determines the columns for filenames and labels in the annotations DataFrame.

        Args:
            filename_col (Optional[str]): Column name for filenames.
            label_col (Optional[str]): Column name for labels.

        Notes:
            if 'filename_col' or 'label_col' is not provided,
            the first and second columns of the annotation file are used.
        """
        df = pd.read_csv(self.annotations_file)
        if filename_col is None:
            self.filename_col = df.columns[0]
        else:
            self.filename_col = filename_col
            assert self.filename_col in df.columns, f"Column not found: {self.filename_col}"

        if label_col is None:
            self.label_col = df.columns[1]
        else:
            self.label_col = label_col
            assert self.label_col in df.columns, f"Column not found: {self.label_col}"
        del df

    def _validate_files(self) -> None:
        """Validates the existence of the image directory and annotations file."""
        assert self.image_dir.exists(), f"Image directory not found: {self.image_dir}"
        assert self.annotations_file.exists(), f"Annotations file not found: {self.annotations_file}"

    def _get_input_size(self, df) -> Tuple[int, int]:
        """Determines the input and output sizes of the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the annotations.

        Returns:
            Tuple[int, int]: A tuple containing the input size and output size.
        """
        ds = ImageDataset(self.image_dir, df, transform=self.transform,
                          target_transform=self.target_transform)
        input, output = next(iter(ds))
        input_size = tuple(input.size())
        if isinstance(output, int):
            output_size = (1,)
        elif isinstance(output, float):
            output_size = (1,)
        elif isinstance(output, Tensor):
            output_size = output.size()
        else:
            raise ValueError(f"Output type not recognized: {type(output)}")
        output_size = tuple(output.size())
        del ds
        return input_size, output_size

    def _extract_df(self) -> None:
        """Extracts essential info from the annotations DataFrame."""
        self.annotations_df = pd.read_csv(self.annotations_file)
        labels = self.annotations_df[self.label_col]
        self.annotations_df = pd.DataFrame(
            {'filename': self.annotations_df[self.filename_col],
             'label': labels,
             'is_train': self.annotations_df['train']})

    def prepare_data(self) -> None:
        """Prepares the data by validating files. Only called once, on only the primary device."""
        self._validate_files()

    def setup(self, stage: str) -> None:
        """Sets up the dataset for training or validation.

        Args:
            stage (str): The stage of the training process ('fit', 'validate', 'test', or 'predict').
        """
        if not hasattr(self, 'annotations_df'):
            self._extract_df()
        if stage == 'fit' or stage is None:
            df = self.annotations_df[self.annotations_df['is_train']][['filename', 'label']]
            df = df.reset_index(drop=True)
            self.input_size, self.output_size = self._get_input_size(df)
            self.train_ds = ImageDataset(self.image_dir, df, transform=self.transform,
                                         target_transform=self.target_transform)

            df = self.annotations_df[~self.annotations_df['is_train']][['filename', 'label']]
            df = df.reset_index(drop=True)
            self.val_ds = ImageDataset(self.image_dir, df, transform=self.transform,
                                       target_transform=self.target_transform)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return NotImplemented
    #
    # def predict_dataloader(self):
    #     return NotImplemented


class TabularDataset(Dataset):
    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]], target):
        super().__init__()
        if isinstance(df, pd.DataFrame):
            self.df = [df]
        else:
            self.df = [_df for _df in df if _df is not None]
        self.target = target
        self._validate_lens()

    def __len__(self):
        return len(self.df[0])

    def _validate_lens(self):
        lens = set([len(df) for df in self.df])
        lens.add(len(self.target))
        assert len(lens) == 1, 'Length of dataframes do not match.'

    def __getitem__(self, idx: int) -> Tuple[FloatTensor, Tensor]:
        row = []
        for df in self.df:
            row.append(df.iloc[idx].values)
        target = torch.tensor(self.target[idx])
        return torch.from_numpy(np.concatenate(row)), target


class TabularDataModule(L.LightningDataModule):
    def __init__(self,
                 df: pd.DataFrame,
                 continuous_cols: list,
                 categorical_cols: list,
                 target_col: str,
                 predict_df: Optional[pd.DataFrame] = None,
                 continuous_scaler: Union[bool, Callable] = True,
                 categorical_encoder=True,
                 target_transform=None,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.df = df
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_workers = num_workers

        if continuous_scaler is True:
            self.continuous_scaler = StandardScaler()
        else:
            self.continuous_scaler = continuous_scaler

        if categorical_encoder is True:
            self.categorical_encoder = self.categorical_encoder = CategoricalEncoder(
                categorical_cols=self.categorical_cols)
        else:
            self.categorical_encoder = categorical_encoder

        self.target_transform = target_transform

        self.predict_df = predict_df

    def prepare_data(self) -> None:
        """Runs checks on the dataframe, and prepares the data for training.

        Only runs on a single host at the time of initialization.
        """
        self._validate_cols('fit')
        if self.predict_df is not None:
            self._validate_cols('predict')

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self._extract_train_val()  # train_df, val_df
            self.train_target = self._target_transform(self.train_df)
            self.val_target = self._target_transform(self.val_df)
            self._encode_categorical(stage)  # train_nominal_df, train_ordinal_df, val_nominal_df, val_ordinal_df
            self._scale_continuous(stage)  # train_cont_df, val_cont_df
            self.train_ds = TabularDataset(df=[self.train_nominal_df, self.train_ordinal_df, self.train_cont_df],
                                           target=self.train_target)
            self.val_ds = TabularDataset(df=[self.val_nominal_df, self.val_ordinal_df, self.val_cont_df],
                                         target=self.val_target)
        else:
            self.target = self._target_transform(self.predict_df)
            self._encode_categorical(stage)  # nominal_df, ordinal_df
            self._scale_continuous(stage)  # cont_df
            self.predict_ds = TabularDataset(df=[self.nominal_df, self.ordinal_df, self.cont_df],
                                             target=self.target)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return NotImplemented
    #
    # def predict_dataloader(self):
    #     return NotImplemented

    def _validate_cols(self, stage: str) -> None:
        """Checks if all required column names are in the dataframe."""
        if stage == 'fit' or stage is None:
            cols = self.df.columns
            assert 'is_train' in cols, "'is_train' column not found in dataframe."
        else:
            cols = self.predict_df.columns
        assert self.continuous_cols in cols, 'Continuous columns not found in dataframe.'
        assert self.categorical_cols in cols, 'Categorical columns not found in dataframe.'
        assert self.target_col in cols, 'Target column not found in dataframe.'

    def _extract_train_val(self):
        """Extracts the training and validation dataframes using 'is_train' column.

        Only required columns are extracted
        """
        required_cols = self.continuous_cols + self.categorical_cols + [self.target_col]
        self.train_df = self.df[self.df['is_train']][required_cols].copy()
        self.val_df = self.df[~self.df['is_train']][required_cols].copy()
        del self.df

    def _target_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_transform is not None:
            target = self.target_transform(df[self.target_col])
        else:
            target = df[self.target_col]
        return target

    def _encode_categorical(self, stage: str) -> None:
        if self.categorical_encoder is not False:
            if stage == 'fit' or stage is None:
                self.train_nominal_df, self.train_ordinal_df = self.categorical_encoder.fit_transform(
                    self.train_df[self.categorical_cols])
                self.val_nominal_df, self.val_ordinal_df = self.categorical_encoder.transform(
                    self.val_df[self.categorical_cols])
            else:
                self.nominal_df, self.ordinal_df = self.categorical_encoder.transform(
                    self.predict_df[self.categorical_cols])
        else:
            if stage == 'fit' or stage is None:
                self.train_nominal_df = self.train_df[self.categorical_cols]
                self.val_nominal_df = self.val_df[self.categorical_cols]
            else:
                self.nominal_df = self.predict_df[self.categorical_cols]

    def _scale_continuous(self, stage: str):
        if self.continuous_scaler is not False:
            if stage == 'fit' or stage is None:
                self.train_cont_df = pd.DataFrame(
                    self.continuous_scaler.fit_transform(self.train_df[self.continuous_cols]),
                    columns=self.continuous_cols, index=self.train_df.index)
                self.val_cont_df = pd.DataFrame(
                    self.continuous_scaler.transform(self.val_df[self.continuous_cols]),
                    columns=self.continuous_cols, index=self.val_df.index)
            else:
                self.cont_df = pd.DataFrame(
                    self.continuous_scaler.transform(self.predict_df[self.continuous_cols]),
                    columns=self.continuous_cols, index=self.predict_df.index)
        else:
            if stage == 'fit' or stage is None:
                self.train_cont_df = self.train_df[self.continuous_cols]
                self.val_cont_df = self.val_df[self.continuous_cols]
            else:
                self.cont_df = self.predict_df[self.continuous_cols]
