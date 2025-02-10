import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", module='mne')

def hash_df(df):
    return sha256(df.to_csv(index=False).encode()).hexdigest()


## FILENAME TAGGER

VERIFIER_PREFIX = 'file_annotations'
VERIFIER_SUFFIX = '.csv'

def filename_tagger(filename: str, verbose=False) -> dict:
    tags = {'dataset': str(filename),
            'balanced': 'N',
            }
    flag, msg = verify_filename(filename)
    if verbose:
        print(msg)
    if flag:
        filename_splits = split_filename(filename)
        if verbose:
            print(filename_splits)
        tagger_funcs = [sex_filter_tagger,
                        age_filter_tagger,
                        bal_filter_tagger,
                        dataset_version_tagger,
                        label_type_tagger]
        for tagger_func in tagger_funcs:
            tags = reduce(lambda acc, txt: tagger_func(acc, txt), filename_splits, tags)
    return tags


def verify_filename(filename):
    if not filename.startswith(VERIFIER_PREFIX):
        return False, 'Filename does not start with the expected prefix.'
    if not filename.endswith(VERIFIER_SUFFIX):
        return False, 'Filename does not end with the expected suffix.'
    return True, 'Filename is valid.'


def split_filename(filename):
    core_name = filename[len(VERIFIER_PREFIX): -len(VERIFIER_SUFFIX)]
    return core_name.split('_')[1:]


def sex_filter_tagger(tags, txt):
    match txt:
        case "m":
            tags.update({'sex_filter': 'm'})
        case "f":
            tags.update({'sex_filter': 'f'})
        case "fm":
            tags.update({'sex_filter': 'all'})
        case "mf":
            tags.update({'sex_filter': 'all'})
    return tags


def age_filter_tagger(tags, txt):
    if txt[-1] == 'y' and '-' in txt:
        age_splits = txt.split('-')
        age_splits[1] = age_splits[1].strip('y')
        if len(age_splits) == 2 and age_splits[0].isdigit() and age_splits[1].isdigit():
            age_limits = [int(age) for age in age_splits]
            tags.update({'age_filter': f'{min(age_limits)}-{max(age_limits)}'})
    return tags


def bal_filter_tagger(tags, txt):
    if txt == 'bal':
        tags.update({'balanced': 'Y'})
    return tags


def dataset_version_tagger(tags, txt):
    if txt[0] == 'v' and txt[1:].isdigit():
        tags.update({'dataset_version': txt})
    return tags

def label_type_tagger(tags, txt):
    match txt:
        case "ald4dx":
            tags.update({'label': 'Alcohol'})
        case "mdd4dx":
            tags.update({'label': 'Depression'})
        case "mdd4dx-all":
            tags.update({'label': 'Depression'})
        case "cod4dx":
            tags.update({'label': 'Cocaine'})
        case "mjd4dx":
            tags.update({'label': 'Marijuana'})
        case "opd4dx":
            tags.update({'label': 'Opioid'})
    return tags

def read_cnt(file_path, data_format='auto', verbose=False):
    try:
        data = mne.io.read_raw_cnt(file_path, preload=True, data_format=data_format, verbose=verbose)
    except Exception as e:
        print(f'Error reading file {file_path}: {e}')
        return None
    return data
## -- ##

class CalculateEEGDist:
    def __init__(self,
                 data_dir: PathLike,
                 save_file: Optional[PathLike] = 'noise_config.json',
                 overwrite: bool = False,
                 reader_func: callable = read_cnt,
                 verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.overwrite = overwrite
        if save_file is not None:
            self.save_file = Path(save_file)
        self.reader_func = reader_func

    @staticmethod
    def get_sample_mean(data):
        if data is not None:
            return data.get_data().mean()

    @staticmethod
    def get_sample_std(data):
        if data is not None:
            return data.get_data().std()

    def task(self, file):
        data = self.reader_func(self.data_dir/file)
        mean = self.get_sample_mean(data)
        std = self.get_sample_std(data)
        del data
        return mean, std

    def calculate(self, annotation_file: PathLike):
        annotation_df = pd.read_csv(annotation_file)
        annotation_df = annotation_df[annotation_df['split'] == 'train']
        total_files = len(annotation_df)
        with tqdm(total=total_files) as pbar:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.task, file) for file in annotation_df.filename]
                means = []
                stds = []
                for future in as_completed(futures):
                    mean, std = future.result()
                    means.append(mean)
                    stds.append(std)
                    pbar.update(1)
        mean = np.mean(means)
        std = np.mean(stds)
        return mean, std

    def check_write_fmt(self):
        file_exitension = self.save_file.suffix
        if file_exitension == '':
            if self.verbose:
                print('Output file does not have a valid extension, will append “.json”.')
            self.save_file = self.save_file.with_suffix('.json')
        elif file_exitension != '.json':
            raise ValueError('Invalid output file extension, must be “.json”.')

    def save(self, annotation_file, mean, std):
        annotation_file = Path(annotation_file)
        output_file = self.data_dir / self.save_file
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                _dict = json.load(f)
        else:
            _dict = {}
        if self.verbose:
            if _dict.get(annotation_file.name) is not None and not self.overwrite:
                print(f'{annotation_file.name} already exists in the output file, skipping save.')
                return
            elif _dict.get(annotation_file.name) is not None and self.overwrite:
                print(f'Overwriting {annotation_file.name} in the output file.')
        _dict[annotation_file.name] = {'mean': mean, 'std': std}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(_dict, f, ensure_ascii=False, indent=4)

    def check_save(self, annotation_file):
        annotation_file = Path(annotation_file)
        output_file = self.data_dir / self.save_file
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                _dict = json.load(f)
            if self.overwrite:
                if self.verbose:
                    if _dict.get(annotation_file.name) is not None:
                        print(f'Overwriting {annotation_file.name} in the output file.')
                return True
            else:
                if _dict.get(annotation_file.name) is not None:
                    if self.verbose:
                        print(f'{annotation_file.name} already exists in the output file, skipping.')
                    return False
                else:
                    return True
        else:
            return True

    def __call__(self, annotation_file: PathLike):
        annotation_file = self.data_dir / annotation_file
        self.check_write_fmt()
        if self.check_save(annotation_file):
            if self.verbose:
                print(f'Calculating mean and std for the dataset, {annotation_file.name}...')
                start_time = time.time()
            mean, std = self.calculate(annotation_file)
            if self.verbose:
                print(f'Mean: {mean}')
                print(f'Std: {std}')
                execution_time = time.time() - start_time
                hours, rem = divmod(execution_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"Execution Time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")
            if self.save_file is not None:
                self.save(annotation_file, mean, std)
            if self.verbose:
                print(f'Saved to {self.save_file}')
