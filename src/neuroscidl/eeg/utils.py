import json
import mne
import numpy as np
import pandas as pd
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from hashlib import sha256
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List, Union

warnings.filterwarnings("ignore", module='mne')

def hash_df(df: pd.DataFrame) -> str:
    """"Returns a SHA256 hash of the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to hash.

    Returns:
        str: The SHA256 hash of the dataframe.
    """
    return sha256(df.to_csv(index=False).encode()).hexdigest()


## FILENAME TAGGER

VERIFIER_PREFIX = 'file_annotations'
VERIFIER_SUFFIX = '.csv'

def filename_tagger(filename: str, verbose=False) -> Dict[str,str]:
    """Tags the filename with relevant metadata.

        Args:
            filename (str): The filename to tag.
            verbose (bool, optional): If True, prints additional information.
                Defaults to False.

        Returns:
            dict: A dictionary of tags extracted from the filename.
        """
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


def verify_filename(filename: str) -> Tuple[bool, str]:
    """Verifies if the filename has the correct prefix and suffix.

        Args:
            filename (str): The filename to verify.

        Returns:
            tuple: A tuple containing a boolean indicating if the filename is valid and a message.
        """
    if not filename.startswith(VERIFIER_PREFIX):
        return False, 'Filename does not start with the expected prefix.'
    if not filename.endswith(VERIFIER_SUFFIX):
        return False, 'Filename does not end with the expected suffix.'
    return True, 'Filename is valid.'


def split_filename(filename: str) -> List[str]:
    """Splits the filename into its components seperated by '_'.

    Args:
        filename (str): The filename to split.

    Returns:
        list: A list of components extracted from the filename.
    """
    core_name = filename[len(VERIFIER_PREFIX): -len(VERIFIER_SUFFIX)]
    return core_name.split('_')[1:]


def sex_filter_tagger(tags: Dict[str, str], txt: str) -> Dict[str, str]:
    """Tagger func: Tags the filename with the sex filter.

    Args:
        tags (dict): The current tags.
        txt (str): The text to tag.

    Returns:
        dict: The updated tags.
    """
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


def age_filter_tagger(tags: Dict[str, str], txt: str) -> Dict[str, str]:
    """Tagger func: Tags the filename with the age filter.

    Args:
        tags (dict): The current tags.
        txt (str): The text to tag.

    Returns:
        dict: The updated tags.
    """
    if txt[-1] == 'y' and '-' in txt:
        age_splits = txt.split('-')
        age_splits[1] = age_splits[1].strip('y')
        if len(age_splits) == 2 and age_splits[0].isdigit() and age_splits[1].isdigit():
            age_limits = [int(age) for age in age_splits]
            tags.update({'age_filter': f'{min(age_limits)}-{max(age_limits)}'})
    return tags


def bal_filter_tagger(tags: Dict[str, str], txt: str) -> Dict[str, str]:
    """Tagger func: Tags the filename with the balance filter.

    Args:
        tags (dict): The current tags.
        txt (str): The text to tag.

    Returns:
        dict: The updated tags.
    """
    if txt == 'bal':
        tags.update({'balanced': 'Y'})
    return tags


def dataset_version_tagger(tags: Dict[str, str], txt: str) -> Dict[str, str]:
    """Tags the filename with the dataset version.

    Args:
        tags (dict): The current tags.
        txt (str): The text to tag.

    Returns:
        dict: The updated tags.
    """
    if txt[0] == 'v' and txt[1:].isdigit():
        tags.update({'dataset_version': txt})
    return tags

def label_type_tagger(tags: Dict[str, str], txt: str) -> Dict[str, str]:
    """Tags the filename with the label type.

    Args:
        tags (dict): The current tags.
        txt (str): The text to tag.

    Returns:
        dict: The updated tags.
    """
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
## -- ##

def read_cnt(file_path: PathLike, preload=True, verbose=False, **kwargs) -> Optional[mne.io.Raw]:
    """Safe reads a CNT file using MNE.

    Args:
        file_path (PathLike): The path to the CNT file.
        preload (bool, optional): Whether to preload the CNT file. Defaults to True.
        verbose (bool, optional): If True, prints additional information.
            Defaults to False.
        **kwargs: Additional arguments passed to mne.io.read_raw_cnt()

    Returns:
        Optional[mne.io.Raw]: The raw data object or None if an error occurs.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f'Path {file_path} is not a file')
    try:
        data = mne.io.read_raw_cnt(file_path, preload=preload, verbose=verbose, **kwargs)
    except Exception as e:
        print(f'Error reading file {file_path}: {e}')
        return None
    return data

def read_fif(file_path: PathLike, preload=True, verbose=False, **kwargs) -> Optional[mne.io.Raw]:
    """Safe reads a FIF file using MNE.

    Args:
        file_path (PathLike): The path to the FIF file.
        preload (bool, optional): Whether to preload the FIF file. Defaults to True.
        verbose (bool, optional): If True, prints additional information.
            Defaults to False.
        **kwargs: Additional arguments passed to mne.io.read_raw_fif()

    Returns:
        Optional[mne.io.Raw]: The raw data object or None if an error occurs.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f'Path {file_path} is not a file')
    try:
        data = mne.io.read_raw_fif(file_path, preload=preload, verbose=verbose, **kwargs)
    except Exception as e:
        print(f'Error reading file {file_path}: {e}')
        return None
    return data

def read_eeg(file_path: PathLike, preload=True, verbose:Union[bool, str, int, None]=False, **kwargs):
    """Reads an EEG file in .fif or .cnt format using MNE.

    Args:
        file_path (PathLike): The path to the EEG file.
        preload (bool, optional): Whether to preload the file. Defaults to True.
        verbose (bool, str, int, optional): Control verbosity of the logging output.
            If None, use the default verbosity level of MNE.
            Defaults to False
        **kwargs: Additional arguments passed to the MNE read functions.

    Returns:
        mne.io.Raw: The raw data object if successful, None otherwise.

    Raises:
        ValueError: If the file path is not a file or the file type is unsupported.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f'Path {file_path} is not a file')
    try:
        file_type = file_path.suffix
        match file_type:
            case '.fif':
                data = mne.io.read_raw_fif(file_path, preload=preload, verbose=verbose, **kwargs)
            case '.cnt':
                data = mne.io.read_raw_cnt(file_path, preload=preload, verbose=verbose, **kwargs)
            case _:
                raise ValueError(f'Unsupported file type: {file_type}. Supported file types are .fif and .cnt')
    except Exception as e:
        print(f'Error reading file {file_path}: {e}')
        return None
    return data

class CalculateEEGDist:
    """Class to calculate the mean and standard deviation of EEG data."""
    def __init__(self,
                 data_dir: PathLike,
                 save_file: Optional[PathLike] = 'noise_config.json',
                 overwrite: bool = False,
                 verbose: bool = True):
        """
        Args:
            data_dir (PathLike): The directory containing the data files.
            save_file (Optional[PathLike], optional): The file to save the results.
                Defaults to 'noise_config.json'.
            overwrite (bool, optional): If True, overwrites existing results.
                Defaults to False.
            verbose (bool, optional): If True, prints additional information.
                Defaults to True.
       """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.overwrite = overwrite
        if save_file is not None:
            self.save_file = Path(save_file)
        self.reader_func = staticmethod(read_eeg)

    @staticmethod
    def get_sample_mean(data: mne.io.Raw) -> Optional[float]:
        """Calculates the mean of the data.

        Args:
            data (mne.io.Raw): The raw data object.

        Returns:
            Optional[float]: The mean of the data or None if data is None.
        """
        if data is not None:
            return data.get_data().mean()

    @staticmethod
    def get_sample_std(data: mne.io.Raw) -> Optional[float]:
        """Calculates the standard deviation of the data.

        Args:
            data (mne.io.Raw): The raw data object.

        Returns:
            Optional[float]: The standard deviation of the data or None if data is None.
        """
        if data is not None:
            return data.get_data().std()

    def task(self, file: PathLike) -> Tuple[float, float]:
        """Reads the data file and calculates the mean and standard deviation.

        Args:
            file (PathLike): The data file to read.

        Returns:
            tuple: A tuple containing the mean and standard deviation of the data.
        """
        data = self.reader_func(self.data_dir/file)
        mean = self.get_sample_mean(data)
        std = self.get_sample_std(data)
        del data
        return mean, std

    def calculate(self, annotation_file: PathLike) -> Tuple[float, float]:
        """Calculates the mean and standard deviation for all files in the annotation file, in parallel.

        Args:
            annotation_file (PathLike): The annotation file containing the list of data files.

        Returns:
            tuple: A tuple containing the overall mean and standard deviation.
        """
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
        return float(mean), float(std)

    def check_write_fmt(self) -> None:
        """Checks and updates the save file format."""
        file_exitension = self.save_file.suffix
        if file_exitension == '':
            if self.verbose:
                print('Output file does not have a valid extension, will append “.json”.')
            self.save_file = self.save_file.with_suffix('.json')
        elif file_exitension != '.json':
            raise ValueError('Invalid output file extension, must be “.json”.')

    def save(self, annotation_file: PathLike, mean: float, std: float) -> None:
        """Saves the mean and standard deviation to the save file.

        Args:
            annotation_file (PathLike): The annotation file.
            mean (float): The mean value.
            std (float): The standard deviation value.
        """
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

    def check_save(self, annotation_file: PathLike) -> bool:
        """Checks if the results should be saved.

        Args:
            annotation_file (PathLike): The annotation file.

        Returns:
            bool: True if the results should be saved, False otherwise.
        """
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

    def __call__(self, annotation_file: PathLike) -> None:
        """Calculates and saves the mean and standard deviation for the dataset in parallel.

        Args:
            annotation_file (PathLike): The annotation file.
        """
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
