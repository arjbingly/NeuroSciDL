from pathlib import Path

import lightning as L
import lightning.pytorch as pl
import torchmetrics as tm
import torch

from pytorch_boilerplate.callbacks import PrintMetricsTableCallback, MlFlowHyperParams, MlFlowModelSummary
from pytorch_boilerplate.eeg_dataset import EEGDataModule
from pytorch_boilerplate.eeg_model import EEGViT_pretrained

# TODO: config file
MODEL_NAME = 'EEGViT'

# Dataset
DATA_DIR = Path('/home/arjbingly/SUNpY/data/eec')
ANNOTATIONS_FILE = DATA_DIR / 'file_annotations.csv'
BATCH_SIZE = 128

# Compute related
ACCELERATOR = "gpu"  # use GPU
# PRECISION = 16  # 16-bit precision
# PROFILER = "simple"  # simple profiler
USE_MLFLOW = True  # use MLFlow for logging

MAX_EPOCHS = 20

datamodule = EEGDataModule(data_dir=DATA_DIR, annotation_file=ANNOTATIONS_FILE, batch_size=BATCH_SIZE, num_workers=10)

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
                }

callbacks = [
    # pl.callbacks.EarlyStopping(monitor='val_loss', patience=6),
     pl.callbacks.ModelCheckpoint(monitor='val_BinaryAccuracy', mode='max', save_top_k=2,
                                  filename=f'{MODEL_NAME}'+'-{epoch:02d}-{val_BinaryAccuracy:.2f}'),
    pl.callbacks.ModelCheckpoint(monitor='val_BinaryF1Score', mode='max', save_top_k=2,
                                 filename=f'{MODEL_NAME}'+'-{epoch:02d}-{val_BinaryF1Score:.2f}'),
     PrintMetricsTableCallback(table_format='pretty', skip_zero_epoch=True, decimal_precision=4),
     MlFlowModelSummary(),
     MlFlowHyperParams(trainer_args),
             ]

torch.set_float32_matmul_precision('high')

if USE_MLFLOW:
    mlf_logger = pl.loggers.MLFlowLogger(log_model='all', tracking_uri='http://localhost:8080')
    # mlf_logger.log_hyperparams(trainer_args)
    trainer_args['logger'] = mlf_logger

trainer = L.Trainer(callbacks=callbacks, **trainer_args)

trainer.fit(model, datamodule)
