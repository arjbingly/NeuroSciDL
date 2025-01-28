import json
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import lightning.pytorch as pl
import torch
import torchmetrics as tm
from neuroscidl.callbacks import PrintMetricsTableCallback, MlFlowHyperParams, MlFlowModelSummary
from neuroscidl.eeg.eeg_dataset import EEGDataModule
from neuroscidl.eeg.eeg_model import EEGViT_pretrained
from neuroscidl.eeg.utils import filename_tagger
from torchvision.transforms.v2 import GaussianNoise

USE_PARSER = True

# Dataset
DATA_DIR = Path('/data/eec')
ANNOTATIONS_FILE = DATA_DIR / 'file_annotations_bal_test.csv' # ignored by parser
BATCH_SIZE = 128

# Model Name
MODEL_PREFIX = 'EEGViT_'

# Noise Aug
NOISE_CONFIG_FILENAME = 'noise_config.json'

# MLFLOW
USE_MLFLOW = True  # use MLFlow for logging
EXPERIMENT_NAME = 'TestBed'
TRACKING_URI = 'http://localhost:8080'

# Compute related
ACCELERATOR = "gpu"  # use GPU
# PRECISION = 16  # 16-bit precision
# PROFILER = "simple"  # simple profiler

MAX_EPOCHS = 20


if USE_PARSER:
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Dataset filename')
    args = parser.parse_args()
    ANNOTATIONS_FILE = DATA_DIR / args.file

# Model Name
ds_name = ANNOTATIONS_FILE.name.replace("file_annotations_", "").split('.')[0]
model_name = f'{MODEL_PREFIX}{ds_name}'

# Check if file annotation exists
if not ANNOTATIONS_FILE.exists():
    raise FileNotFoundError(f'Annotation File {ANNOTATIONS_FILE} does not exist')

# Noise Aug
if not (DATA_DIR/NOISE_CONFIG_FILENAME).exists():
    raise FileNotFoundError(f'Noise Config File {DATA_DIR/NOISE_CONFIG_FILENAME} does not exist')
with open(DATA_DIR/NOISE_CONFIG_FILENAME, 'r') as f:
    noise_config = json.load(f)

noise_config = noise_config[Path(ANNOTATIONS_FILE).name]
experimental_mean = noise_config['mean']
experimental_std = noise_config['std']

gaussian_mean = round(experimental_mean, 6)
gaussian_std = round(experimental_std, 6) * 0.5

gaussian_noise_transform = GaussianNoise(mean=gaussian_mean, sigma=gaussian_std, clip=False)

datamodule = EEGDataModule(data_dir=DATA_DIR, annotation_file=ANNOTATIONS_FILE, batch_size=BATCH_SIZE, num_workers=10, train_transform=gaussian_noise_transform)

# train model
metrics = [tm.classification.F1Score(task='binary', average='macro'),
           tm.classification.Accuracy(task='binary'),
           tm.classification.Precision(task='binary'),
           tm.classification.Recall(task='binary'), ]

model = EEGViT_pretrained(metrics=metrics)

trainer_args = {'max_epochs': MAX_EPOCHS,
                'check_val_every_n_epoch': 1,
                # 'gpus': 1,
                # 'progress_bar_refresh_rate': 20,
                # 'precision': 16,
                # 'profiler': PROFILER,
                # 'auto_lr_find': True,
                # 'auto_scale_batch_size': 'binsearch',
                # 'num_sanity_val_steps': 4,

                # DEBUG OPTIONS
                # overfit_batches=1
                # 'fast_dev_run': True,
                # 'limit_train_batches': 0.1,
                # 'limit_val_batches': 0.1,

                # NOISE
                'gaussian_mean': gaussian_mean,
                'gaussian_std': gaussian_std,

                # DATASET
                'data_dir': DATA_DIR,
                'annotation_file': ANNOTATIONS_FILE,
                'batch_size': BATCH_SIZE,

                }

callbacks = [
    # pl.callbacks.EarlyStopping(monitor='val_loss', patience=6),
    pl.callbacks.ModelCheckpoint(monitor='val_BinaryAccuracy', mode='max', save_top_k=2,
                                 filename=f'{model_name}' + '-{epoch:02d}-{val_BinaryAccuracy:.2f}'),
    pl.callbacks.ModelCheckpoint(monitor='val_BinaryF1Score', mode='max', save_top_k=2,
                                 filename=f'{model_name}' + '-{epoch:02d}-{val_BinaryF1Score:.2f}'),
    PrintMetricsTableCallback(table_format='pretty', skip_zero_epoch=True, decimal_precision=4),
    MlFlowModelSummary(),
    MlFlowHyperParams(trainer_args),
]

torch.set_float32_matmul_precision('high')

tags = filename_tagger(ANNOTATIONS_FILE.name)
print(f'Tags: {tags}')

if USE_MLFLOW:
    mlf_logger = pl.loggers.MLFlowLogger(log_model='all',experiment_name=EXPERIMENT_NAME, tracking_uri=TRACKING_URI, tags=tags)
    # mlf_logger.log_hyperparams(trainer_args)
    trainer_args['logger'] = mlf_logger

trainer = L.Trainer(callbacks=callbacks, **trainer_args)

trainer.fit(model, datamodule)
