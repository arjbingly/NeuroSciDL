from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir: Union[str, Path], annotations_df: pd.DataFrame, transform=None, target_transform=None
                 # TODO: type hint for transform
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.annotations_df = annotations_df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.annotations_df)

    def __getitem__(self, idx: int) -> Tuple[Image, int]:
        img_path = self.image_dir / self.annotations_df.iloc[idx, 0]
        label = self.annotations_df.iloc[idx, 1]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class ImageDataModule(L.LightningDataModule):
    def __init__(self, image_dir: Union[str, Path], annotations_file: Union[str, Path], batch_size: int = 32,
                 filename_col: Optional[str] = None, label_col: Optional[str] = None, transform=None,
                 # TODO: type hint for transform
                 target_transform=None,  # TODO: type hint for transform
                 num_workers: int = 0):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.annotations_file = annotations_file
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
        assert self.image_dir.exists(), f"Image directory not found: {self.image_dir}"
        assert self.annotations_file.exists(), f"Annotations file not found: {self.annotations_file}"

    def _get_input_size(self, df) -> Tuple[int, int]:
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
        self.annotations_df = pd.read_csv(self.annotations_file)
        labels = self.annotations_df[self.label_col]
        self.annotations_df = pd.DataFrame(
            {'filename': self.annotations_df[self.filename_col],
             'label': labels,
             'is_train': self.annotations_df['train']})

    def prepare_data(self) -> None:
        self._validate_files()

    def setup(self, stage: str) -> None:
        if not hasattr(self, 'annotations_df'):
            self._extract_df()
        # self._extract_df()
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
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return NotImplemented
    #
    # def predict_dataloader(self):
    #     return NotImplemented
