from functools import reduce
from hashlib import sha256


def hash_df(df):
    return sha256(df.to_csv(index=False).encode()).hexdigest()


## FILENAME TAGGER

VERIFIER_PREFIX = 'file_annotations'
VERIFIER_SUFFIX = '.csv'


def filename_tagger(filename: str, verbose=False) -> dict:
    tags = {'dataset': filename,
            'balanced': 'False',}
    flag, msg = verify_filename(filename)
    if verbose:
        print(msg)
    if flag:
        filename_splits = split_filename(filename)
        if verbose:
            print(filename_splits)
        tagger_funcs = [sex_filter_tagger, age_filter_tagger, bal_filter_tagger, dataset_version_tagger]
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
        tags.update({'balanced': 'True'})
    return tags


def dataset_version_tagger(tags, txt):
    if txt[0] == 'v' and txt[1:].isdigit():
        tags.update({'dataset version': txt[1:]})
    return tags
