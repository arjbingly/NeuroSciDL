import copy
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Mapping

import lightning.pytorch as pl
import torch
import torchinfo
from jsonargparse import namespace_to_dict, Namespace
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import SaveConfigCallback
from notifiers.providers.pushover import Pushover
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

    def __init__(self, mlflow_params: Dict, other_params:Optional[Dict]=None,log_callbacks=True, log_optimizers=True) -> None:
        self.params = mlflow_params
        if other_params is not None:
            self.params.update(other_params)
        self.replaceable_keys = ['logger', 'profiler', 'plugins']
        self.log_callbacks = log_callbacks
        self.log_optimizers = log_optimizers
        self.model_checkpoint_callbacks = {}
        self.model_checkpoint_params = {}

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
        checkpoint_counter = 0
        if hasattr(trainer, 'callbacks'):
            for i, c in enumerate(trainer.callbacks):
                if hasattr(c, 'state_dict'):
                    callback_name = c.__class__.__qualname__
                    if callback_name == 'ModelCheckpoint':
                        _callback_name = f'ModelCheckpoint_{checkpoint_counter}'
                        self.model_checkpoint_callbacks[_callback_name] = c
                        checkpoint_counter += 1
                    else:
                        callback_params = c.state_dict()
                        for p in callback_params:
                            self.params[str(callback_name) + '-' + p] = callback_params[p]

    def get_model_checkpoint_info(self, trainer: "pl.Trainer") -> None:
        """Logs model checkpoint information to params."""
        for callback_name, callback in self.model_checkpoint_callbacks.items():
            callback_params = callback.state_dict()
            for p in callback_params:
                if 'best_k_models' not in p:
                    self.model_checkpoint_params[str(callback_name) + '-' + p] = callback_params[p]


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


    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
        """
        self.get_model_checkpoint_info(trainer)
        trainer.logger.log_hyperparams(self.model_checkpoint_params)

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
        metric_categories (Optional[Mapping[str, str]]): The mapping of metric prefixes to category names.
            Defaults to {"train_": “Train Metrics”, "val_": “Val Metrics”}.
        table_format (str, optional): The format of the table. Defaults to 'pretty'.
            Refer to the tabulate documentation for all possible values.
        skip_zero_epoch (bool, optional): Whether to skip printing the table for the zero epoch. Defaults to True.
        decimal_precision (int, optional): The decimal precision for the metrics. Defaults to 4.

    Notes:
        Uses tabulate to convert the metrics dictionary to a table string. For the detailed information on behaviour
        see the tabulate documentation: https://github.com/astanin/python-tabulate
    """

    def __init__(self,
                 metric_categories: Optional[Mapping[str, str]] = None,
                 table_format: str = 'pretty',
                 skip_zero_epoch: bool = True,
                 decimal_precision: int = 4) -> None:
        self.metrics: Dict = {}
        self.metric_categories: Optional[Mapping[str, str]] = metric_categories or {"train_": "Train Metrics", "val_": "Val Metrics"}
        self.category_metrics: Dict[str,Dict] = {category: {} for category in self.metric_categories.values()}
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
        if self.skip_zero_epoch and trainer.current_epoch == 0:
            return

        self.metrics = copy.copy(trainer.callback_metrics)
        self.metrics = self.format_dict(self.metrics)
        self.separate_metrics_by_category()
        tables = [self.metrics_to_table(self.category_metrics[category], headers=['Metric', 'Score']) for category in self.metric_categories.values()]
        table = self.sidebyside_tables(tables, headers=list(self.category_metrics.keys()))
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

    @staticmethod
    def format_table(table: str) -> str:
        """Formats the table string.

        Args:
            table (str): The table string.

        Returns:
            str: The formatted table string.
        """
        return '\n' + table

    def separate_metrics_by_category(self):
        """Separates the metrics into categories."""
        for metric_name, metric_score in self.metrics.items():
            for prefix, category in self.metric_categories.items():
                if metric_name.startswith(prefix):
                    self.category_metrics[category][metric_name.removeprefix(prefix)] = metric_score

    def metrics_to_table(self, dict: Dict, headers: List[str]) -> str:
        """Converts the metrics dictionary to a table string.

        Args:
            dict (Dict): The dictionary of metrics.
            headers (List[str]): The headers for the table.

        Returns:
            str: The table string.
        """
        return tabulate(dict.items(), headers=headers, tablefmt=self.table_format, numalign='decimal')

    def sidebyside_tables(self, tables:List[str], headers: List[str]) -> str:
        """Combines two tables side by side.

        Args:
            tables (List[str]): The list of table strings.
            headers (List[str]): The headers for the combined table.

        Returns:
            str: The combined table string.
        """
        table_lines = [table.splitlines() for table in tables]
        return tabulate([list(item) for item in zip(*table_lines)], headers=headers, tablefmt=self.table_format)

# This class is necessary due to a bug in the current implementation of the SaveConfigCallback when used in conjunction with MLFlowLogger
# Refer to: https://github.com/Lightning-AI/pytorch-lightning/issues/16310
# Solution based on:
#   https://github.com/Lightning-AI/pytorch-lightning/issues/16310#issuecomment-2371750960
#   https://github.com/Lightning-AI/pytorch-lightning/issues/16310#issuecomment-1980538782

class MLFlowSaveConfigCallback(SaveConfigCallback):
    """A callback to save the configuration file and log it to MLFlow.

        Args:
            parser (LightningArgumentParser): The argument parser used to parse the configuration.
            config (Namespace): The configuration namespace.
            config_filename (str, optional): The name of the configuration file. Defaults to 'config.yaml'.
            overwrite (bool, optional): Whether to overwrite the existing configuration file. Defaults to False.
            multifile (bool, optional): Whether to save the configuration in multiple files. Defaults to False.
            store_artifact (bool, optional): Whether to store the configuration as an artifact in MLFlow. Defaults to True.
            log_hyperparams (bool, optional): Whether to log the configuration as hyperparameters in MLFlow. Defaults to True.
        """
    def __init__(self, parser, config, config_filename='config.yaml', overwrite=False, multifile=False, store_artifact=True, log_hyperparams=True):
        super().__init__(parser, config, config_filename, overwrite, multifile, save_to_log_dir=False)
        self.store_artifact = store_artifact
        self.log_hyperparams = log_hyperparams

    def config_to_log(self, config: Namespace) -> Dict:
        """Converts the configuration namespace to a dictionary for hyperparameter logging.

        Args:
            config (Namespace): The configuration namespace.

        Returns:
            Dict: The configuration dictionary for logging.
        """
        config_dict = namespace_to_dict(config)
        config_dict = self.convert_list_to_dict(config_dict)

        return config_dict

    def convert_list_to_dict(self, config: Dict) -> Dict:
        """Recursively converts lists in the configuration dictionary to a more readable format.

        Args:
            config (Dict): The configuration dictionary.

        Returns:
            Dict: The converted configuration dictionary.
        """
        new_config = {}
        for key, value in config.items():
            if isinstance(value, list):
                new_config[key] = {i: self.convert_list_to_dict(v) if isinstance(v, dict) else v for i, v in
                                   enumerate(value)}
            elif isinstance(value, dict):
                new_config[key] = self.convert_list_to_dict(value)
            else:
                new_config[key] = value
        return new_config

            Dict: The converted configuration dictionary.
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Saves the configuration and logs it to MLFlow.

         Args:
             trainer (Trainer): The trainer instance.
             pl_module (LightningModule): The LightningModule instance.
             stage (str): The stage of the training process.
         """
        # Convert Namespace to dict
        config_dict = vars(self.config)

        if self.log_hyperparams:
            # Convert Namespace to loggable dict
            config_dict = self.config_to_log(self.config)
            # Log parameters to MLFlow
            pl_module.logger.log_hyperparams(config_dict)

        if self.store_artifact:
            # Log artifact, save as yaml
            if trainer.is_global_zero:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    config_path = Path(tmp_dir) / 'config.yaml'
                    self.parser.save(
                        self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
                    )
                    trainer.logger.experiment.log_artifact(local_path=config_path,
                                                           run_id=trainer.logger.run_id)

class NotifyCallback(Callback):
    """A callback to send notifications using PushOver when training starts and ends.
    The callback requires the environment variables NOTIFIERS_PUSHOVER_USER and NOTIFIERS_PUSHOVER_TOKEN to be set.

    Args:
        send_start (bool, optional): Whether to send a notification when training starts. Defaults to True.
        send_end (bool, optional): Whether to send a notification when training ends. Defaults to True.
        verbose (bool, optional): Whether to print the notification details. Defaults to True.
    """

    def __init__(self, send_start: bool = True, send_end:bool = True, verbose=True) -> None:
        self.send_start = send_start
        self.send_end = send_end
        self.notifier = Pushover()
        self.user_key_var = 'NOTIFIERS_PUSHOVER_USER'
        self.token_var = 'NOTIFIERS_PUSHOVER_TOKEN'
        self.verbose = verbose
        self.notified_start = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when the training stage is set up.
        Checks for user_id and token in the environment variables.
        Extracts the run_url from the trainer logger.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
            stage (str): The stage of the training process.
        """
        if stage == 'fit':
            self.check_env_vars()
        self.run_url = self.get_run_url(trainer)

    def check_env_vars(self):
        """Checks if the environment variables for user_id and token are set."""
        if os.environ.get(self.user_key_var) is None or os.environ.get(self.token_var) is None:
            raise ValueError(f'Environment variables {self.user_key_var} and {self.token_var} must be set.')

    def get_run_url(self, trainer: "pl.Trainer") -> str:
        """Generates the run URL from the trainer logger."""
        return f'{trainer.logger._tracking_uri}/#/experiments/{trainer.logger.experiment_id}/runs/{trainer.logger.run_id}'

    def get_run_info(self, trainer: "pl.Trainer") -> str:
        """Gets the run information from the trainer logger.
        Returns:
            mlflow.entities.RunInfo: The run information.
            It contains the following attributes:
            - artifact_uri
            - end_time
            - experiment_id
            - lifecycle_stage
            - run_id
            - run_name
            - run_uuid
            - start_time
            - status
            - user_id
        """
        return trainer.logger.experiment.get_run(trainer.logger.run_id).info

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit starts.
        It sends a notification if send_start is True.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
        """
        if self.send_start:
            info = self.get_run_info(trainer)
            msg_title = 'Training Started 🚀'
            msg_body = f'Training run {info.run_name} has started.\nStarted on: {datetime.fromtimestamp(info.start_time/1000).strftime('%a %d %b %Y, %I:%M%p')}'
            self.notifier.notify(title=msg_title, message=msg_body, url=self.run_url, url_title='View Run')
            self.notified_start = True
            if self.verbose:
                print('Notification sent.')
                print('Message Title: ',msg_title)
                print('Message Body: ',msg_body)
                print('Run URL: ',self.run_url)


    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Called when an exception occurs.
        It sends a notification of a failed run if the initial start notification was sent.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
            exception (BaseException): The exception that occurred.
        """
        if self.notified_start:
            info = self.get_run_info(trainer)
            run_duration = datetime.now() - datetime.fromtimestamp(info.start_time / 1000)
            msg_title = 'Training Failed 🚨'
            msg_body = f'Training run {info.run_name} has failed due to the following exception: {exception}\nApprox. Duration: {str(run_duration)}'
            self.notifier.notify(title=msg_title, message=msg_body, url=self.run_url, url_title='View Run')
            if self.verbose:
                print('Notification sent.')
                print('Message Title: ',msg_title)
                print('Message Body: ',msg_body)
                print('Run URL: ', self.run_url)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends.
        It sends a notification if send_end is True.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
        """
        if self.send_end:
            info = self.get_run_info(trainer)
            if info.end_time is not None:
                run_duration = datetime.fromtimestamp(info.end_time / 1000) - datetime.fromtimestamp(info.start_time / 1000)
            else:
                run_duration = datetime.now() - datetime.fromtimestamp(info.start_time / 1000)
            msg_title = 'Training Failed 🚨'
            msg_body = f'Training run {info.run_name} has failed. \n Approx. Duration: {str(run_duration)}.'
            if info.status == 'RUNNING':
                msg_title = 'Training Completed 🎉'
                msg_body = f'Training run {trainer.logger.name} has completed successfully. \nApprox. Duration: {str(run_duration)}.'
            self.notifier.notify(title=msg_title, message=msg_body, url=self.run_url, url_title='View Run')
            if self.verbose:
                print('Notification sent.')
                print('Message Title: ',msg_title)
                print('Message Body: ',msg_body)
                print('Run URL: ', self.run_url)
