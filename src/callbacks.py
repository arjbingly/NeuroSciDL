import copy
import tempfile
from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
import torchinfo
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from tabulate import tabulate


class MlFlowHyperParams(Callback):
    def __init__(self, params: Dict, log_callbacks=True, log_optimizers=True) -> None:
        self.params = params
        self.replaceable_keys = ['logger', 'profiler', 'plugins']
        self.log_callbacks = log_callbacks
        self.log_optimizers = log_optimizers

    def replace_with_state_key(self, key: str) -> None:
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

    def get_optimizer_info(self, trainer: "pl.Trainer") -> Dict[str, str]:
        optimizers = trainer.optimizers
        for i, optimizer in enumerate(optimizers):
            self.params[f'optimizer_{i}'] = str(optimizer)

    def get_callbacks_info(self, trainer: "pl.Trainer") -> Dict[str, str]:
        if hasattr(trainer, 'callbacks'):
            for i, c in enumerate(trainer.callbacks):
                if hasattr(c, 'state_dict'):
                    callback_name = c.__class__.__qualname__
                    callback_params = c.state_dict()
                    for p in callback_params:
                        self.params[str(callback_name) + '-' + p] = callback_params[p]

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_callbacks:
            self.get_callbacks_info(trainer)
        if self.log_optimizers:
            self.get_optimizer_info(trainer)
        for key in self.replaceable_keys:
            self.replace_with_state_key(key)
        trainer.logger.log_hyperparams(self.params)


class MlFlowModelSummary(Callback):
    def __init__(self,
                 input_size=None,
                 input_data=None,
                 output_size=None,
                 batch_dim=None,
                 device=None,
                 dtypes=None,
                 log_artifact=True,
                 log_summary_as_param=False) -> None:
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
        return self._generate_state_key(log_artifact=self.log_artifact, log_summary_as_param=self.log_summary_as_param)

    def log_summary_artifact(self, trainer: "pl.Trainer", summary: str) -> None:
        run_id = trainer.logger.run_id
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "summary.txt")
            path.write_text(summary)
            trainer.logger.experiment.log_artifact(run_id, path, "model")

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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
    """Prints a table with the metrics in columns on every epoch end."""

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
        return self._generate_state_key(table_format=self.table_format,
                                        skip_zero_epoch=self.skip_zero_epoch,
                                        decimal_precision=self.decimal_precision)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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
        _dict = {}
        for key, value in dict.items():
            if key not in self.skip_metrics:
                _dict[key] = round(float(value), self.decimal_precision)
        return _dict

    def format_table(self, table: str) -> str:
        return '\n' + table

    def seperate_train_val_metrics(self):
        for metric_name, metric_score in self.metrics.items():
            if metric_name.startswith(self.val_prefix):
                self.val_metrics[metric_name.removeprefix(self.val_prefix)] = metric_score
            elif metric_name.startswith(self.train_prefix):
                self.train_metrics[metric_name.removeprefix(self.train_prefix)] = metric_score

    def metrics_to_table(self, dict: Dict, headers: List[str]) -> str:
        return tabulate(dict.items(), headers=headers, tablefmt=self.table_format, numalign='decimal')

    def sidebyside_table(self, table1: str, table2: str, headers: List[str]) -> str:
        table1_lines = table1.splitlines()
        table2_lines = table2.splitlines()
        return tabulate([list(item) for item in zip(table1_lines, table2_lines)], headers=headers,
                        tablefmt=self.table_format)
