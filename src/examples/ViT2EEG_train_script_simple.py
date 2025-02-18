from pathlib import Path
from argparse import ArgumentParser
import lightning as L
import lightning.pytorch as pl
import torchmetrics as tm
import torch
import json

from imposter_logger import NUM_WORKERS
from torchvision.transforms.v2 import GaussianNoise

from neuroscidl.callbacks import PrintMetricsTableCallback, MlFlowHyperParams, MlFlowModelSummary
from neuroscidl.eeg.eeg_dataset import EEGDataModule
from neuroscidl.eeg.eeg_model import EEGViT_pretrained

DEV_RUN = True

# Dataset
DATA_DIR = Path('/data/eec')
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
ANNOTATIONS_FILE = ANNOTATIONS_DIR / 'file_annotations.csv'
BATCH_SIZE = 128

# Model Name
MODEL_PREFIX = 'EEGViT_'

# Noise Aug
NOISE_CONFIG_FILENAME = 'noise_config.json'

# Model Name
ds_name = ANNOTATIONS_FILE.name.replace("file_annotations_", "").split('.')[0]
model_name = f'{MODEL_PREFIX}{ds_name}'

# Compute related
ACCELERATOR = "gpu"  # use GPU
# PRECISION = 16  # 16-bit precision
# PROFILER = "simple"  # simple profiler
USE_MLFLOW = False  # use MLFlow for logging

MAX_EPOCHS = 2

NUM_WORKERS_DATAMODULE = 10


# Check if file annotation exists
if not ANNOTATIONS_FILE.exists():
    raise FileNotFoundError(f'Annotation File {ANNOTATIONS_FILE} does not exist')

# Noise Aug
with open(DATA_DIR/NOISE_CONFIG_FILENAME, 'r') as f:
    noise_config = json.load(f)

noise_config = noise_config[ANNOTATIONS_FILE.name]
experimental_mean = noise_config['mean']
experimental_std = noise_config['std']

gaussian_mean = round(experimental_mean, 6)
gaussian_std = round(experimental_std, 6) * 0.5

gaussian_noise_transform = GaussianNoise(mean=gaussian_mean, sigma=gaussian_std, clip=False)

datamodule = EEGDataModule(data_dir=DATA_DIR, annotation_file=ANNOTATIONS_FILE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATAMODULE, train_transform=gaussian_noise_transform)

# train model
metrics = [tm.classification.F1Score(task='binary', average='macro'),
           tm.classification.Accuracy(task='binary'),
           tm.classification.Precision(task='binary'),
           tm.classification.Recall(task='binary'), ]

model = EEGViT_pretrained(metrics=metrics)

trainer_args = {'max_epochs': MAX_EPOCHS,
                'check_val_every_n_epoch': 1,
                }

if DEV_RUN:
    trainer_args['limit_train_batches'] = 2
    trainer_args['limit_val_batches'] = 2

callbacks = [
    # pl.callbacks.EarlyStopping(monitor='val_loss', patience=6),
     pl.callbacks.ModelCheckpoint(monitor='val_BinaryAccuracy', mode='max', save_top_k=2,
                                  filename=f'{model_name}' + '-{epoch:02d}-{val_BinaryAccuracy:.2f}'),
    pl.callbacks.ModelCheckpoint(monitor='val_BinaryF1Score', mode='max', save_top_k=2,
                                 filename=f'{model_name}' + '-{epoch:02d}-{val_BinaryF1Score:.2f}'),
     PrintMetricsTableCallback(table_format='pretty', skip_zero_epoch=True, decimal_precision=4),
             ]

torch.set_float32_matmul_precision('high')

if USE_MLFLOW:
    callbacks.extend([MlFlowModelSummary(),
                      MlFlowHyperParams(trainer_args)])
    mlf_logger = pl.loggers.MLFlowLogger(log_model='all', tracking_uri='http://localhost:8080')
    # mlf_logger.log_hyperparams(trainer_args)
    trainer_args['logger'] = mlf_logger

trainer = L.Trainer(callbacks=callbacks, **trainer_args)

trainer.fit(model, datamodule)
