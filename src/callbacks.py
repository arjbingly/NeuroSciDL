import copy
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import lightning.pytorch as pl
import torch
import torchinfo
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from tabulate import tabulate
from torchinfo.torchinfo import INPUT_DATA_TYPE, INPUT_SIZE_TYPE


class MlFlowHyperParams(Callback):
    """Logs hyperparameters to MlFlow.

    Args:
        params (Dict): Dictionary of hyperparameters to log, typically trainer config.
        log_callbacks (bool, optional): Whether to log callback parameters. Defaults to True.
        log_optimizers (bool, optional): Whether to log optimizer parameters. Defaults to True.
    """

    def __init__(self, params: Dict, log_callbacks=True, log_optimizers=True) -> None:
        self.params = params
        self.replaceable_keys = ['logger', 'profiler', 'plugins']
        self.log_callbacks = log_callbacks
        self.log_optimizers = log_optimizers

    def replace_with_state_key(self, key: str) -> None:
        """Replaces the given key in params with its state key if it exists.

        Args:
            key (str): The key to replace in the params dictionary.
        """
        if key in self.params:
            if isinstance(self.params[key], List):
                _items = []
                for item in self.params[key]:
                    if hasattr(item, 'state_key'):
                        _items.append(item.state_key)
                    else:
                        _items.append(item)
                self.params[key] = _items
            else:
                if hasattr(self.params[key], 'state_key'):
                    self.params[key] = self.params[key].state_key

    def get_optimizer_info(self, trainer: pl.Trainer) -> None:
        """Logs optimizer information to params.

        Args:
            trainer (pl.Trainer): The trainer instance.
        """
        optimizers = trainer.optimizers
        for i, optimizer in enumerate(optimizers):
            self.params[f'optimizer_{i}'] = str(optimizer)

    def get_callbacks_info(self, trainer: "pl.Trainer") -> None:
        """Logs callback information to params.

        Args:
            trainer (pl.Trainer): The trainer instance.
        """
        if hasattr(trainer, 'callbacks'):
            for i, c in enumerate(trainer.callbacks):
                if hasattr(c, 'state_dict'):
                    callback_name = c.__class__.__qualname__
                    callback_params = c.state_dict()
                    for p in callback_params:
                        self.params[str(callback_name) + '-' + p] = callback_params[p]

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit starts.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
        """
        if self.log_callbacks:
            self.get_callbacks_info(trainer)
        if self.log_optimizers:
            self.get_optimizer_info(trainer)
        for key in self.replaceable_keys:
            self.replace_with_state_key(key)
        trainer.logger.log_hyperparams(self.params)


class MlFlowModelSummary(Callback):
    """Logs model summary to MlFlow.

    Args:
        input_size (list/tuple/torch.size, optional): The input size of the model.
            Default will try to get from the model.
        input_data (sequence of tensors, optional): The input data for the model.
            Default will try to get from the model.
        output_size (list/tuple/torch.size, optional): The output size of the model.
        batch_dim (int, optional): The batch dimension of input data.
            If None, assume input_data / input_size contains the batch dimension
        device (torch.Device, optional): The device to use for model and input data.
        dtypes (List[torch.dtype], optional): The data types to use.
            If you use input_size, torchinfo assumes your input uses FloatTensors.
            If your model use a different data type, specify that dtype.
        log_artifact (bool, optional): Whether to log the summary as an artifact. Defaults to True.
        log_summary_as_param (bool, optional): Whether to log the summary as a parameter. Defaults to False.

    Notes:
        This uses the torchinfo.summary function to get the model summary. For more detailed information on behavior
        see the torchinfo documentation: https://github.com/TylerYep/torchinfo?tab=readme-ov-file#documentation
        If neither input_data or input_size are provided, no forward pass through the
        network is performed, and the provided model information is limited to layer names.
    """

    def __init__(self,
                 input_size: Optional[INPUT_SIZE_TYPE] = None,
                 input_data: Optional[INPUT_DATA_TYPE] = None,
                 output_size: Optional[INPUT_SIZE_TYPE] = None,
                 batch_dim: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 dtypes: Optional[List[torch.dtype]] = None,
                 log_artifact: bool = True,
                 log_summary_as_param: bool = False) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.input_data = input_data
        self.batch_dim = batch_dim
        self.device = device
        self.dtypes = dtypes
        self.log_artifact = log_artifact
        self.log_summary_as_param = log_summary_as_param

    @property
    def state_key(self) -> str:
        """Generates a state key for the callback.

        Returns:
            str: The state key.
        """
        return self._generate_state_key(log_artifact=self.log_artifact, log_summary_as_param=self.log_summary_as_param)

    def log_summary_artifact(self, trainer: "pl.Trainer", summary: str) -> None:
        """Logs the model summary as an artifact.

        Args:
            trainer (pl.Trainer): The trainer instance.
            summary (str): The model summary.
        """
        run_id = trainer.logger.run_id
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "summary.txt")
            path.write_text(summary)
            trainer.logger.experiment.log_artifact(run_id, path, "model")

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit starts.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
        """
        if self.input_size is None:
            if hasattr(pl_module, 'input_size'):
                self.input_size = pl_module.input_size
        if self.output_size is None:
            if hasattr(pl_module, 'output_size'):
                self.output_size = pl_module.output_size

        model_summary = torchinfo.summary(pl_module,
                                          input_size=self.input_size,
                                          input_data=self.input_data,
                                          batch_dim=self.batch_dim,
                                          device=self.device,
                                          dtypes=self.dtypes,
                                          verbose=0)
        if self.log_summary_as_param:
            trainer.logger.log_hyperparams({'model_summary': str(model_summary)})
        other_info = {'total_params': model_summary.total_params,
                      'trainable_params': model_summary.trainable_params}

        if self.input_size is not None:
            other_info['input_size'] = self.input_size
        if self.input_data is not None:
            other_info['input_data'] = self.input_data
        if self.output_size is not None:
            other_info['output_size'] = self.output_size
        if self.batch_dim is not None:
            other_info['batch_dim'] = self.batch_dim
        if self.device is not None:
            other_info['device'] = self.device
        if self.dtypes is not None:
            other_info['dtypes'] = self.dtypes

        if hasattr(trainer.model, 'criterion'):
            other_info['criterion'] = str(trainer.model.criterion)

        trainer.logger.log_hyperparams(other_info)

        if self.log_artifact:
            self.log_summary_artifact(trainer, str(model_summary))


class PrintMetricsTableCallback(Callback):
    """Prints a table with the metrics in columns on every epoch end.

    Args:
        table_format (str, optional): The format of the table. Defaults to 'pretty'.
            Refer to the tabulate documentation for all possible values.
        skip_zero_epoch (bool, optional): Whether to skip printing the table for the zero epoch. Defaults to True.
        decimal_precision (int, optional): The decimal precision for the metrics. Defaults to 4.

    Notes:
        Uses tabulate to convert the metrics dictionary to a table string. For the detailed information on behaviour
        see the tabulate documentation: https://github.com/astanin/python-tabulate
    """

    def __init__(self, table_format: str = 'pretty', skip_zero_epoch: bool = True, decimal_precision: int = 4) -> None:
        self.metrics: Dict = {}
        self.val_prefix: str = "val_"
        self.train_prefix: str = "train_"
        self.val_metrics: Dict = {}
        self.train_metrics: Dict = {}
        self.table_format: str = table_format
        self.skip_zero_epoch = skip_zero_epoch
        self.decimal_precision = decimal_precision
        self.skip_metrics = ['train_loss_step', 'train_loss_epoch', 'val_loss_step', 'val_loss_epoch']

    @property
    def state_key(self) -> str:
        """Generates a state key for the callback.

        Returns:
            str: The state key.
        """
        return self._generate_state_key(table_format=self.table_format,
                                        skip_zero_epoch=self.skip_zero_epoch,
                                        decimal_precision=self.decimal_precision)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the training epoch ends.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
        """
        if self.skip_zero_epoch:
            if trainer.current_epoch == 0:
                return

        self.metrics = copy.copy(trainer.callback_metrics)
        self.metrics = self.format_dict(self.metrics)
        self.seperate_train_val_metrics()
        table_headers = ['Metric', 'Score']
        val_metrics_table = self.metrics_to_table(self.val_metrics, table_headers)
        train_metrics_table = self.metrics_to_table(self.train_metrics, table_headers)
        table = self.sidebyside_table(train_metrics_table, val_metrics_table, headers=['Train Metrics', 'Val Metrics'])
        rank_zero_info(self.format_table(table))

    def format_dict(self, dict: Dict) -> Dict:
        """Formats the dictionary of metrics.

        Args:
            dict (Dict): The dictionary of metrics.

        Returns:
            Dict: The formatted dictionary.
        """
        _dict = {}
        for key, value in dict.items():
            if key not in self.skip_metrics:
                _dict[key] = round(float(value), self.decimal_precision)
        return _dict

    def format_table(self, table: str) -> str:
        """Formats the table string.

        Args:
            table (str): The table string.

        Returns:
            str: The formatted table string.
        """
        return '\n' + table

    def seperate_train_val_metrics(self):
        """Separates the training and validation metrics."""
        for metric_name, metric_score in self.metrics.items():
            if metric_name.startswith(self.val_prefix):
                self.val_metrics[metric_name.removeprefix(self.val_prefix)] = metric_score
            elif metric_name.startswith(self.train_prefix):
                self.train_metrics[metric_name.removeprefix(self.train_prefix)] = metric_score

    def metrics_to_table(self, dict: Dict, headers: List[str]) -> str:
        """Converts the metrics dictionary to a table string.

        Args:
            dict (Dict): The dictionary of metrics.
            headers (List[str]): The headers for the table.

        Returns:
            str: The table string.
        """
        return tabulate(dict.items(), headers=headers, tablefmt=self.table_format, numalign='decimal')

    def sidebyside_table(self, table1: str, table2: str, headers: List[str]) -> str:
        """Combines two tables side by side.

        Args:
            table1 (str): The first table string.
            table2 (str): The second table string.
            headers (List[str]): The headers for the combined table.

        Returns:
            str: The combined table string.
        """
        table1_lines = table1.splitlines()
        table2_lines = table2.splitlines()
        return tabulate([list(item) for item in zip(table1_lines, table2_lines)], headers=headers,
                        tablefmt=self.table_format)
