import json
import warnings
from os import PathLike
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from neuroscidl.eeg.utils import hash_df

warnings.filterwarnings("ignore", module='mne')

class CNTSampleAnnotator:
    """
    A class to annotate windowed EEG samples, given a dataframe with column 'n_times'
    corresponding to the total number of datapoints.
    Note:
        file_annotations should contain a column 'n_times' with the number of samples for each file.
        argument file_annotations can be a path to a csv file or a dataframe,
        but the path is preferred due to caching and filename resolution.

    Attributes:
        window_size (int): The size of the window for each sample.
        window_stride (int): The stride of the window for each sample.
        window_start (int): The starting index for the window.
        save_path (Path): The path to save the annotations.
        save_suffix (str): The suffix to add to the saved annotation files.
        overwrite (bool): Whether to overwrite existing annotation files.
        file_annotations (pd.DataFrame): The DataFrame containing file annotations.
        save_filename (str): The filename to save the annotations.
    """
    def __init__(self,
                 file_annotations: Union[pd.DataFrame, PathLike],
                 window_size: int= 500,
                 window_stride: int = 500,
                 window_start: int = 0,
                 save_path: Union[str, Path] = '.',
                 save_suffix: str = 'sample_annotations',
                 overwrite: bool = False,):
        """
        Initializes the CNTSampleAnnotator with the given parameters.

        Args:
            file_annotations (Union[pd.DataFrame, PathLike]): The path to the CSV file containing the file annotations
            or the DataFrame itself, path is preferred since the filename is used to generate the save_filename and cache.
            window_size (int): The size of the window for each sample.
            window_stride (int): The stride of the window for each sample.
            window_start (int): The starting index for the window.
            save_path (Union[str, Path]): The path to save the annotations and config.
            save_suffix (str): The suffix to add to the saved annotation files.
            If a DataFrame is passed as file_annotations, the save_filename will be set to this.
            overwrite (bool): Whether to overwrite existing annotation files, cached files will not be used.
        """
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_start = window_start
        self.save_path = Path(save_path)
        self.save_suffix = save_suffix
        self.overwrite = overwrite
        if isinstance(file_annotations, PathLike):
            file_path = Path(file_annotations)
            self.file_annotations = pd.read_csv(file_path)
            self.save_filename = '_'.join([file_path.stem, self.save_suffix])
        else:
            self.file_annotations = file_annotations
            self.save_filename = self.save_suffix

    def get_config(self) -> Dict[str, Union[int, str]]:
        """
        Returns the configuration of the annotator.

        Returns:
            dict: The configuration of the annotator.
        """
        return {
            'window_size': self.window_size,
            'window_stride': self.window_stride,
            'window_start': self.window_start,
            'file_annotations_hash': hash_df(self.file_annotations),
        }

    @property
    def config(self) -> Dict[str, Union[int, str]]:
        """
        Property to get the configuration of the annotator.

        Returns:
            dict: The configuration of the annotator.
        """
        return self.get_config()

    def save_config(self) -> None:
        """
        Saves the configuration of the annotator to a JSON file.
        The JSON file is saved to the save_path as a hidden file with the same name as the annotation file.
        """
        with open(self.save_path/f'.{self.save_filename}.json', 'w') as f:
            json.dump(self.config, f)

    def get_sample_indexes(self, n_samples: int) -> List[Tuple[int,int]]:
        """
        Returns the start and end indexes for each sample window.

        Args:
            n_samples (int): The total number of samples.

        Returns:
            list: A list of tuples containing the start and end indexes for each sample window.
        """
        start_indices = np.arange(self.window_start, n_samples - self.window_size, self.window_stride)
        end_indices = start_indices + self.window_size
        return list(zip(start_indices.tolist(), end_indices.tolist()))

    def get_sample_info(self) -> pd.DataFrame:
        """
        Generates and returns a DataFrame containing the sample information for each window, sample annotations.

        Returns:
            pd.DataFrame: A DataFrame containing the sample information for each window.
        """
        sample_indices = map(self.get_sample_indexes, self.file_annotations['n_times'])
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

    def get_cached_annotations(self) -> Optional[pd.DataFrame]:
        """
        Returns the cached annotations if they exist and match the current configuration.

        Returns:
            pd.DataFrame: The cached annotations if they exist and match the current configuration, otherwise None.
        """
        sample_annotations_config = '.'+self.save_filename + '.json'
        sample_annotations_filename = self.save_filename + '.csv'
        if (self.save_path/sample_annotations_config).exists() and (self.save_path/sample_annotations_filename).exists():
            with open(self.save_path/sample_annotations_config, 'r') as f:
                config = json.load(f)
                if config == self.config:
                    return pd.read_csv(self.save_path/sample_annotations_filename)

    def __call__(self) -> pd.DataFrame:
        """
        Reads or generates and returns the sample annotations.

        Returns:
            pd.DataFrame: The sample annotations.
        """
        if not self.overwrite:
            sample_annotations = self.get_cached_annotations()
            if sample_annotations is not None:
                print('Found cached sample annotations.')
                return sample_annotations
        self.sample_annotations = self.get_sample_info()
        self.sample_annotations.to_csv(self.save_path/f'{self.save_filename}.csv', index=False)
        self.save_config()
        print('Saved sample annotations.')
        return self.sample_annotations
