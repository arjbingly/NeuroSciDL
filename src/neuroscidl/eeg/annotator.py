import json
import warnings
from os import PathLike
from pathlib import Path
from typing import Union

import mne
import numpy as np
import pandas as pd
from neuroscidl.eeg.utils import hash_df

warnings.filterwarnings("ignore", module='mne')

class CNTSampleAnnotator:
    def __init__(self,
                 file_annotations: Union[pd.DataFrame, PathLike],
                 window_size: int= 500,
                 window_stride: int = 500,
                 window_start: int = 0,
                 save_path: Union[str, Path] = '.',
                 save_suffix: str = 'sample_annotations',):
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_start = window_start
        self.save_path = Path(save_path)
        self.save_suffix = save_suffix
        if isinstance(file_annotations, PathLike):
            file_path = Path(file_annotations)
            self.file_annotations = pd.read_csv(file_path)
            self.save_filename = '_'.join([file_path.stem, self.save_suffix])
        else:
            self.file_annotations = file_annotations
            self.save_filename = self.save_suffix


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
            'window_start': self.window_start,
            'file_annotations_hash': hash_df(self.file_annotations),
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
        return self.sample_annotations
