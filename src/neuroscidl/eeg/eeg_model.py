import torch
from torch import nn
import transformers
import logging
from typing import List, Optional, Union

import lightning as L
import torch.nn as nn
import torchmetrics as tm
from torch import Tensor
from torch.optim import Adam

logger = logging.getLogger(__name__)

class _EEGViT_pretrained(nn.Module):
    def __init__(self, model_params=None):
        super().__init__()
        self.model_params = {
        'conv_out_channels' : 256,
        'conv_kernel_size' : (1, 36),
        'conv_stride' : (1, 36),
        'conv_padding' : (0,2),
        'conv_bias' : False,
        'img_size' : (64,14),
        'patch_size' : (8,1),
        'vit_model_name' : "google/vit-base-patch16-224",
        'embedding_dim' : 768,
        'hidden_size' : 1000,
        'dropout_rate' : 0.1}
        if model_params is not None:
            self.model_params.update(model_params)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.model_params['conv_out_channels'],
            kernel_size=self.model_params['conv_kernel_size'],
            stride=self.model_params['conv_stride'],
            padding=self.model_params['conv_padding'],
            bias=self.model_params['conv_bias']
        )
        self.batchnorm1 = nn.BatchNorm2d(self.model_params['conv_out_channels'], False)
        config = transformers.ViTConfig.from_pretrained(self.model_params['vit_model_name'])
        config.update({'num_channels': self.model_params['conv_out_channels']})
        config.update({'image_size': self.model_params['img_size']})
        config.update({'patch_size': self.model_params['patch_size']})

        model = transformers.ViTForImageClassification.from_pretrained(self.model_params['vit_model_name'], config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(self.model_params['conv_out_channels'],
                                                                           self.model_params['embedding_dim'],
                                                                           kernel_size=self.model_params['patch_size'],
                                                                           stride=self.model_params['patch_size'],
                                                                           padding=(0,0),
                                                                           groups=self.model_params['conv_out_channels'])
        model.classifier=torch.nn.Sequential(torch.nn.Linear(self.model_params['embedding_dim'],self.model_params['hidden_size'],bias=True),
                                             torch.nn.Dropout(p=self.model_params['dropout_rate']),
                                             torch.nn.Linear(self.model_params['hidden_size'],1,bias=True))
        self.ViT = model

    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        return x

class EEGViT_pretrained(L.LightningModule):
    def __init__(self, trainable_base: bool=True, output_dim: int = 1, model_params=None, metrics: Union[None, List[tm.Metric], tm.MetricCollection] = None,
                 criterion: Optional[nn.Module] = None, lr=1e-3):
        super().__init__()
        self.model = _EEGViT_pretrained(model_params)
        if not trainable_base:
            self.model.eval()
        if criterion is None:
            if output_dim == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
                logger.warning('Warning: CrossEntropyLoss used for multi-class classification')
        else:
            self.criterion = criterion
        self.metrics = metrics
        self.lr = lr

    def setup(self, stage: str) -> None:
        """Sets up the metrics for training and validation.

        Args:
            stage (str): The stage of the training process ('fit', 'validate', 'test', or 'predict').
        """
        if not isinstance(self.metrics, tm.MetricCollection):
            self.metrics = tm.MetricCollection(self.metrics)
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.metrics.reset()
        self.val_metrics.reset()
        self.train_metrics.reset()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = self.criterion(x, y)
        return loss, x, y

    def print_metrics(self, scores:dict, prefix:str):
        for key, value in scores.items():
            self.log(f'{prefix}_{key}', value)


    def training_step(self, batch, batch_idx):
        loss, x, y = self._step(batch, batch_idx)
        self.train_metrics.update( x, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch to compute and log metrics."""
        self.train_scores = self.train_metrics.compute()
        self.log_dict(self.train_scores)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, x, y = self._step(batch, batch_idx)
        self.val_metrics.update(x, y)
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch to compute and log metrics."""
        self.val_scores = self.val_metrics.compute()
        self.log_dict(self.val_scores)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Defines the test step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
        loss, x, y = self._step(batch, batch_idx)
        self.metrics.update(x, y)
        self.log('test_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch to compute and log metrics."""
        self.scores = self.metrics.compute()
        self.log_dict(self.scores)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,},
        }
