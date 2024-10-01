from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
