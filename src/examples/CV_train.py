from pathlib import Path

import lightning as L
import lightning.pytorch as pl
import torch
import torchmetrics as tm
from torchvision import models

from pytorch_boilerplate.callbacks import (
    MlFlowHyperParams,
    MlFlowModelSummary,
    PrintMetricsTableCallback,
)
from pytorch_boilerplate.dataset import ImageDataModule
from pytorch_boilerplate.model import PreTrainedImgClassifier
from pytorch_boilerplate.transforms import preprocess_resnet

# ...Params...
# Dataset
DATA_DIR = Path('../../Data/resnet2')
ANNOTATIONS_FILE = DATA_DIR / 'resnet_balanced_df.csv'
BATCH_SIZE = 32
FILENAME_COL = 'file'
LABEL_COL = 'label'
# Compute related
ACCELERATOR = "gpu"  # use GPU
PRECISION = 16  # 16-bit precision
PROFILER = "simple"  # simple profiler
USE_MLFLOW = True  # use MLFlow for logging
# ..............

datamodule = ImageDataModule(image_dir=DATA_DIR,
                             annotation_file=ANNOTATIONS_FILE,
                             batch_size=BATCH_SIZE,
                             filename_col=FILENAME_COL,
                             label_col=LABEL_COL,
                             transform=preprocess_resnet,
                             target_transform=lambda x: torch.tensor(x).unsqueeze(0).float())

# train model
base_model = models.resnet50(pretrained=True)
output_dim = 1
trainable_base = False
metrics = [tm.classification.F1Score(task='binary', average='macro'),
           tm.classification.Accuracy(task='binary'),
           # tm.classification.ConfusionMatrix(task="binary", num_classes=2, normalize='true'),
           tm.classification.Precision(task='binary'),
           tm.classification.Recall(task='binary'), ]

model = PreTrainedImgClassifier(base_model, output_dim, trainable_base, metrics)

trainer_args = {'max_epochs': 2,
                # 'gpus': 1,
                # 'progress_bar_refresh_rate': 20,
                'precision': 16,
                'profiler': 'simple',
                # 'logger': mlf_logger,
                # 'auto_lr_find': True,
                # 'auto_scale_batch_size': 'binsearch',
                # 'num_sanity_val_steps': 4,

                # DEBUG OPTIONS
                # overfit_batches=1
                # 'fast_dev_run': True,
                'limit_train_batches': 0.1,
                'limit_val_batches': 0.1,
                }

callbacks = [pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             pl.callbacks.ModelCheckpoint(monitor='val_BinaryAccuracy', mode='max', save_top_k=2,
                                          filename='resnet50-{epoch:02d}-{val_BinaryAccuracy:.2f}'),
             PrintMetricsTableCallback(table_format='pretty', skip_zero_epoch=True, decimal_precision=4),
             MlFlowModelSummary(),
             MlFlowHyperParams(trainer_args),
             # pl.callbacks.ModelSummary()
             # pl.callbacks.RichModelSummary(),
             # pl.callbacks.RichProgressBar()
             ]

if USE_MLFLOW:
    mlf_logger = pl.loggers.MLFlowLogger(log_model='all', tracking_uri='http://localhost:8080')
    # mlf_logger.log_hyperparams(trainer_args)
    trainer_args['logger'] = mlf_logger

trainer = L.Trainer(callbacks=callbacks, **trainer_args)

# def print_auto_logged_info(r):
#     tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
#     print(f"run_id: {r.info.run_id}")
#     print(f"artifacts: {artifacts}")
#     print(f"params: {r.data.params}")
#     print(f"metrics: {r.data.metrics}")
#     print(f"tags: {tags}")

# if USE_MLFLOW:
#     trainer.fit(model, datamodule)
#         # print_auto_logged_info(run)
#     # trainer.fit(model, datamodule)

trainer.fit(model, datamodule)

# trainer.tune(model, train_loader) #hyperparameter tuning
# trainer.fit(model, datamodule)
# trainer.validate(model, datamodule)
# trainer.test(model, datamodule)
