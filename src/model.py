import logging
from typing import List, Optional, Union

import lightning as L
import torch.nn as nn
import torchmetrics as tm
from torch import Tensor
from torch.optim import Adam

logger = logging.getLogger(__name__)


class PreTrainedImgClassifier(L.LightningModule):
    """A LightningModule for image classification using a pre-trained model.

    Args:
        base_model (Union[nn.Module, L.LightningModule]): The pre-trained base model.
        output_dim (int): The dimension of the output layer.
        trainable_base (bool, optional): Whether to make the base model trainable. Defaults to False.
        metrics (Union[None, List[tm.Metric], tm.MetricCollection], optional): Metrics to evaluate the model. Defaults to None.
        criterion (Optional[nn.Module], optional): Loss function. If None, defaults to BCEWithLogitsLoss for binary classification
            and CrossEntropyLoss for multi-class classification. Defaults to None.
    """

    def __init__(self,
                 base_model: Union[nn.Module, L.LightningModule],
                 output_dim: int,
                 trainable_base: bool = False,
                 metrics: Union[None, List[tm.Metric], tm.MetricCollection] = None,
                 criterion: Optional[nn.Module] = None,
                 # transform: Callable = None
                 ):
        super().__init__()
        self.base_model = base_model
        # self.base_model.trainable = trainable_base
        num_filters = base_model.fc.in_features
        layers = list(base_model.children())[:-1]
        if not trainable_base:
            self.base_model.eval()
        self.fc1 = nn.Linear(self.base_model.fc.out_features, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_dim)
        if criterion is None:
            if output_dim == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
                logger.warning('Warning: CrossEntropyLoss used for multi-class classification')
        else:
            self.criterion = criterion
        self.metrics = metrics
        # self.transform = transform

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

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _step(self, batch, batch_idx):
        """Performs a single step in training, validation, or testing.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            Tuple: A tuple containing the loss, predictions, and targets.
        """
        x, y = batch
        x = self.forward(x)
        loss = self.criterion(x, y)
        return loss, x, y

    def print_metrics(self, scores: dict) -> None:
        """Prints the computed metrics.

        Args:
            scores (dict): Dictionary of metric names and values.
        """
        for k, v in scores.items():
            print(f'{k}: {v.numpy()}')

    def training_step(self, batch, batch_idx):
        """Defines the training step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
        loss, x, y = self._step(batch, batch_idx)
        self.train_metrics.update(x, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch to compute and log metrics."""
        self.train_scores = self.train_metrics.compute()
        self.log_dict(self.train_scores)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """Defines the validation step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
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
        """Configures the optimizer for training.

        Returns:
            Optimizer: The optimizer instance.
        """
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
