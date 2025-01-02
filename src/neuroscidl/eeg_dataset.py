from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from typing import Union
import mne
import numpy as np
import warnings
import json

warnings.filterwarnings("ignore", module='mne')

class CNTSampleAnnotator:
    def __init__(self,
                 file_annotations: pd.DataFrame,
                 window_size: int= 500,
                 window_stride: int = 500,
                 window_start: int = 0,
                 save_path: Union[str, Path] = '.',
                 save_filename: str = 'sample_annotations',):
        self.file_annotations = file_annotations
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_start = window_start
        self.save_path = Path(save_path)
        self.save_filename = save_filename

    @classmethod
    def read_cnt(file_path: Union[str, Path], data_format='auto', verbose=False):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError(f'Path {file_path} is not a file')
        try:
            data = mne.io.read_raw_cnt(file_path, preload=True, data_format=data_format, verbose=verbose)
        except Exception as e:
            print(f'Error reading file {file_path}: {e}')
            return None
        return data

    def get_config(self):
        return {
            'window_size': self.window_size,
            'window_stride': self.window_stride,
            'window_start': self.window_start
        }

    @property
    def config(self):
        return self.get_config()

    def save_config(self):
        with open(self.save_path/f'{self.save_filename}_config.json', 'w') as f:
            json.dump(self.config, f)

    def get_sample_indices(self, n_samples):
        start_indices = np.arange(self.window_start, n_samples - self.window_size, self.window_stride)
        end_indices = start_indices + self.window_size
        return list(zip(list(start_indices), list(end_indices)))

    def get_sample_info(self):
        sample_indices = map(self.get_sample_indices, self.file_annotations['n_times'])
        records = []
        for i, idx  in enumerate(sample_indices):
            _record = self.file_annotations.iloc[i].to_dict()
            for j, (start_idx, stop_idx) in enumerate(idx):
                record = _record.copy()
                record['sample_id'] = j
                record['start_idx'] = start_idx
                record['stop_idx'] = stop_idx
                records.append(record)
        return pd.DataFrame.from_records(records)

    def __call__(self, *args, **kwargs):
        self.sample_annotations = self.get_sample_info()
        self.sample_annotations.to_csv(self.save_path/f'{self.save_filename}.csv', index=False)
        self.save_config()

class EEGSampleDataset(Dataset):
    def __init__(self, data_dir: Union[str, Path], annotations_df: pd.DataFrame,transform=None, online_transforms=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.annotations_df = annotations_df
        self.transform = transform
        self.online_transforms = online_transforms

    def __len__(self):
        return len(self.annotations_df)

    @classmethod
    def read_cnt(cls, file_path: Union[str, Path], data_format='auto', verbose=False):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError(f'Path {file_path} is not a file')
        data = mne.io.read_raw_cnt(file_path, data_format=data_format, verbose=verbose)
        return data

    def pad_channels(self, data, pad_channels:int):
        return np.pad(data, ((0,pad_channels),(0,0)), mode='constant', constant_values=0)

    def __getitem__(self, idx: int):
        file_path = self.data_dir / self.annotations_df.iloc[idx, 0]
        label = self.annotations_df.iloc[idx]['label'].astype(np.float32)
        start_idx = self.annotations_df.iloc[idx]['start_idx']
        stop_idx = self.annotations_df.iloc[idx]['stop_idx']
        raw = self.read_cnt(file_path)
        data = raw.get_data(start=start_idx, stop=stop_idx)
        if self.annotations_df.iloc[idx]['n_channels'] == 32:
            data = self.pad_channels(data, 32)
        data = np.expand_dims(data, axis=0).astype(np.float32)
        return torch.from_numpy(data), torch.tensor([label])

class EEGDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Union[str, Path],
                 annotation_file: Union[str, Path],
                 batch_size: int,
                 # TODO: Custom annotations columns
                 online_transforms=None,
                 offline_transforms=None,
                 target_transform=None,
                 window_config=(500,500,0), # window_size, window_stride, window_start,
                 num_workers: int = 0,
                 ):

        super().__init__()
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file)
        self.annotations_df = None
        self.batch_size = batch_size
        self.online_transforms = online_transforms
        self.offline_transforms = offline_transforms
        self.target_transform = target_transform
        self.window_config = window_config
        self.num_workers = num_workers

    def _validate_files(self) -> None:
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"
        assert self.annotation_file.exists(), f"Annotation file not found: {self.annotation_file}"

    def _validate_annotations(self) -> None:
        assert 'filename' in self.annotations_df.columns, f"Column 'filename' not found in annotation file"
        assert 'n_times' in self.annotations_df.columns, f"Column 'n_times' not found in annotation file"
        assert 'freq' in self.annotations_df.columns, f"Column 'freq' not found in annotation file"
        assert 'n_channels' in self.annotations_df.columns, f"Column 'n_channels' not found in annotation file"
        assert 'label' in self.annotations_df.columns, f"Column 'label' not found in annotation file"
        for col in ['filename', 'n_times', 'freq', 'n_channels', 'label']:
            assert self.annotations_df[col].notnull().all(), f"Column '{col}' has missing values in annotation file"
        for file in self.annotations_df['filename']:
            assert (self.data_dir/file).exists(), f"File {file} not found in {self.data_dir}"

    def prepare_data(self, stage=None):
        self._validate_files()
        self.annotations_df = pd.read_csv(self.annotation_file)
        self._validate_annotations()

    def _annotate_samples(self):
        if (self.data_dir/'sample_annotations_config.json').exists() and (self.data_dir/'sample_annotations.csv').exists():
            with open(self.data_dir/'sample_annotations_config.json', 'r') as f:
                config = json.load(f)
                # TODO: add a hash of both annotations file in the config
                if config['window_size'] == self.window_config[0] and \
                   config['window_stride'] == self.window_config[1] and \
                   config['window_start'] == self.window_config[2]:
                    return pd.read_csv(self.data_dir/'sample_annotations.csv')

        self.annotator = CNTSampleAnnotator(self.annotations_df, *self.window_config, save_path=self.data_dir, save_filename='sample_annotations')
        self.annotator()
        return pd.read_csv(self.data_dir/'sample_annotations.csv')

    def setup(self, stage=None):
        self.annotations_df = self._annotate_samples()
        if stage == 'fit' or stage is None:
            self.train_annotations = self.annotations_df[self.annotations_df['split']=='train']
            self.val_annotations = self.annotations_df[self.annotations_df['split']=='val']
        if stage == 'test' or stage is None:
            self.test_annotations = self.annotations_df[self.annotations_df['split']=='test']

    def train_dataloader(self):
        return DataLoader(
            EEGSampleDataset(self.data_dir, self.train_annotations, transform=self.offline_transforms, online_transforms=self.online_transforms),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            EEGSampleDataset(self.data_dir, self.val_annotations, transform=self.offline_transforms, online_transforms=self.online_transforms),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            EEGSampleDataset(self.data_dir, self.test_annotations, transform=self.offline_transforms, online_transforms=self.online_transforms),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False)
