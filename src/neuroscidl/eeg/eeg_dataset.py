from pathlib import Path
from typing import Union

import mne
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from neuroscidl.eeg.annotator import CNTSampleAnnotator
from torch.utils.data import DataLoader, Dataset


class EEGSampleDataset(Dataset):
    def __init__(self,
                 data_dir: Union[str, Path],
                 annotations_df: pd.DataFrame,
                 transform=None,
                 target_transform=None,
                 return_info=None,
                 label_col: str='label'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.annotations_df = annotations_df
        self.return_info = return_info
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_df)

    @classmethod
    def read_cnt(cls, file_path: Union[str, Path], data_format='auto', verbose=False):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError(f'Path {file_path} is not a file')
        data = mne.io.read_raw_cnt(file_path, data_format=data_format, verbose=verbose)
        return data

    @staticmethod
    def pad_channels(data, pad_channels:int):
        return np.pad(data, ((0,pad_channels),(0,0)), mode='constant', constant_values=0)

    @staticmethod
    def apply_transform(data, transform):
        if transform is None:
            return data
        elif callable(transform):
            return  transform(data)
        elif isinstance(transform, list):
            for t in transform:
                data = t(data)
            return data
        else:
            raise TypeError(f'Transform should be callable or list of callables, but got {type(transform)}')

    def __getitem__(self, idx: int):
        file_path = self.data_dir / self.annotations_df.iloc[idx, 0]
        label = self.annotations_df.iloc[idx][self.label_col].astype(np.float32)
        start_idx = self.annotations_df.iloc[idx]['start_idx']
        stop_idx = self.annotations_df.iloc[idx]['stop_idx']
        raw = self.read_cnt(file_path)
        data = raw.get_data(start=start_idx, stop=stop_idx)
        if self.annotations_df.iloc[idx]['n_channels'] == 32:
            data = self.pad_channels(data, 32)
        data = np.expand_dims(data, axis=0).astype(np.float32)

        data = self.apply_transform(data, self.transform)
        label = self.apply_transform(label, self.target_transform)

        if self.return_info is None:
            return torch.from_numpy(data), torch.tensor([label])
        else:
            info = self.annotations_df.iloc[idx][self.return_info]
            return torch.from_numpy(data), torch.tensor([label]), info

class EEGDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Union[str, Path],
                 annotation_dir: Union[str, Path],
                 annotation_file: Union[str, Path],
                 batch_size: int,
                 # TODO: Custom annotations columns
                 train_transform=None,
                 val_transform=None,
                 target_transform=None,
                 window_config=(500,500,0),  # window_size, window_stride, window_start,
                 num_workers: int = 0,
                 label_col='label'
                 ):

        super().__init__()
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir)
        self.annotation_file = self.annotation_dir / Path(annotation_file)
        self.annotations_df = None
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        self.window_config = window_config
        self.num_workers = num_workers
        self.label_col = label_col

    def _validate_files(self) -> None:
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"
        assert self.annotation_file.exists(), f"Annotation file not found: {self.annotation_file}"

    def _validate_annotations(self) -> None:
        assert 'filename' in self.annotations_df.columns, "Column 'filename' not found in annotation file"
        assert 'n_times' in self.annotations_df.columns, "Column 'n_times' not found in annotation file"
        assert 'freq' in self.annotations_df.columns, "Column 'freq' not found in annotation file"
        assert 'n_channels' in self.annotations_df.columns, "Column 'n_channels' not found in annotation file"
        assert self.label_col in self.annotations_df.columns, f"Column {self.label_col} not found in annotation file"
        for col in ['filename', 'n_times', 'freq', 'n_channels', self.label_col]:
            assert self.annotations_df[col].notnull().all(), f"Column '{col}' has missing values in annotation file"
        for file in self.annotations_df['filename']:
            assert (self.data_dir/file).exists(), f"File {file} not found in {self.data_dir}"

    def prepare_data(self, stage=None):
        self._validate_files()
        self.annotations_df = pd.read_csv(self.annotation_file)
        self._validate_annotations()

    def _annotate_samples(self):
        self.annotator = CNTSampleAnnotator(self.annotation_file,
                                            *self.window_config,
                                            save_path=self.annotation_dir,
                                            save_suffix='sample_annotations',
                                            overwrite=False)
        return self.annotator()

    def setup(self, stage=None):
        self.annotations_df = self._annotate_samples()
        if stage == 'fit' or stage is None:
            self.train_annotations = self.annotations_df[self.annotations_df['split']=='train']
            self.val_annotations = self.annotations_df[self.annotations_df['split']=='val']
        if stage == 'validate' or stage is None:
            self.val_annotations = self.annotations_df[self.annotations_df['split']=='val']
        if stage == 'test':
            self.test_annotations = self.annotations_df[self.annotations_df['split']=='test']

    def train_dataloader(self):
        return DataLoader(
            EEGSampleDataset(self.data_dir, self.train_annotations, transform=self.train_transform, label_col=self.label_col),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            EEGSampleDataset(self.data_dir, self.val_annotations, transform=self.val_transform, return_info='id', label_col=self.label_col),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            EEGSampleDataset(self.data_dir, self.test_annotations, transform=self.val_transform,return_info='id', label_col=self.label_col),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False)
