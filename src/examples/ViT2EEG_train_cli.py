"""CLI Interface for training EEGViT model.

Example:
    python ViT2EEG_train_cli.py -c config.yaml fit --dev_run --data.annotation_file file_annotations.csv
"""
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from neuroscidl.callbacks import MLFlowSaveConfigCallback
from neuroscidl.eeg.eeg_dataset import EEGDataModule
from neuroscidl.eeg.eeg_model import EEGViT_pretrained
from neuroscidl.eeg.utils import filename_tagger, CalculateEEGDist


class MyLightningCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the parser.
        model_prefix and dev_run are added to the parser.

        Args:
            parser (LightningArgumentParser): The argument parser to add arguments to.
        """
        parser.add_argument('--model_prefix', type=str, default='EEGViT')
        parser.add_argument('--dev_run', action='store_true', help='Run a development run')
        # parser.link_arguments('data.annotation_file', 'data.train_transform.init_args.config_key')

    def dev_run(self):
        """Set the config to run a development run."""
        print('Running a development run....')
        self.config['fit']['trainer']['max_epochs'] = 2
        self.config['fit']['trainer']['limit_train_batches'] = 10
        self.config['fit']['trainer']['limit_val_batches'] = 10
        self.config['fit']['trainer']['logger']['init_args']['experiment_name'] = 'TestBed'
        for i, cl in enumerate(self.config['fit']['trainer']['callbacks']):
            if 'neuroscidl.callbacks.NotifyCallback' in str(cl):
                del self.config['fit']['trainer']['callbacks'][i]

    def update_modelcheckpoint_callback(self):
        """Update the ModelCheckpoint callback filename."""
        ds_name = Path(self.config['fit']['data']['annotation_file']).name.replace("file_annotations_", "").split('.')[0]
        for i, cb in enumerate(self.config['fit']['trainer']['callbacks']):
            if 'ModelCheckpoint' in cb['class_path']:
                self.config['fit']['trainer']['callbacks'][i]['init_args']['filename'] = \
                    (f"{self.config['fit']['model_prefix']}_{ds_name}" +
                     '-{epoch:02d}-{' + f"{self.config['fit']['trainer']['callbacks'][i]['init_args']['monitor']}" +':.2f}')

    def update_mlflow_tags(self):
        """Update the MLFlow logger tags."""
        print('Updating MLFlow logger tags...')
        self.config['fit']['trainer']['logger']['init_args']['tags'] = filename_tagger(
            Path(self.config['fit']['data']['annotation_file']).name)
        print(
            f"{Path(self.config['fit']['data']['annotation_file']).name}:{self.config['fit']['trainer']['logger']['init_args']['tags']}")

    def modify_transform_config(self, transform_config, modify=True):
        """Modify the transform config to use the same config_key as the annotation_file."""
        if transform_config.get('init_args') is not None:
            if transform_config['init_args'].get('config_key') is not None:
                if transform_config['init_args']['config_key'] != self.config['fit']['data']['annotation_file']:
                    if modify:
                        print(
                            f'Warning: {transform_config['class_path']} config_key is not the same as annotation_file, will be updated.')
                        transform_config['init_args']['config_key'] = self.config['fit']['data']['annotation_file']
                    else:
                        print(
                            f'Warning: {transform_config['class_path']} config_key is not the same as annotation_file.')

    def verify_transform_config_key(self, modify=True):
        """Validate train_transform config_key is same as annotation_file."""
        if isinstance(self.config['fit']['data']['train_transform'], list):
            for transform in self.config['fit']['data']['train_transform']:
                self.modify_transform_config(transform, modify)
        else:
            self.modify_transform_config(self.config['fit']['data']['train_transform'], modify)

    def find_noise_config(self):
        """Checks if config for file exists in noise_config.json else create."""
        noise_config_keys = []
        if isinstance(self.config['fit']['data']['train_transform'], list):
            for transform in self.config['fit']['data']['train_transform']:
                if transform.get('init_args') is not None:
                    if transform['init_args'].get('config_key') is not None:
                        noise_config_keys.append(transform['init_args']['config_key'])
        else:
            if self.config['fit']['data']['train_transform'].get('init_args') is not None:
                if self.config['fit']['data']['train_transform']['init_args'].get('config_key') is not None:
                    noise_config_keys.append(self.config['fit']['data']['train_transform']['init_args']['config_key'])

        print('Checking noise config...')
        for noise_config_key in noise_config_keys:
            calculator = CalculateEEGDist(self.config['fit']['data']['data_dir'])
            calculator(noise_config_key)

    def before_instantiate_classes(self) -> None:
        """Modify the configuration before instantiating classes."""
        # Define dev_run
        if self.config['fit'].get('dev_run'):
            self.dev_run()

        # Update ModelCheckpoint callback filename
        self.update_modelcheckpoint_callback()

        # Update MLFlow logger tags
        if self.config['fit']['trainer'].get('logger') is not None:
            if str(self.config['fit']['trainer']['logger']['class_path']) == 'lightning.pytorch.loggers.MLFlowLogger':
                self.update_mlflow_tags()

        # Transforms config
        if self.config['fit']['data'].get('train_transform') is not None:
            # Validate train_transform config_key is same as annotation_file
            self.verify_transform_config_key(modify=True)
            # Checks if config for file exists in noise_config.json else create
            self.find_noise_config()

def cli_main():
    """Main function to run the CLI."""
    torch.set_float32_matmul_precision('high')
    cli = MyLightningCli(EEGViT_pretrained,
                         EEGDataModule,
                         save_config_callback=MLFlowSaveConfigCallback,
                         parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == '__main__':
    cli_main()
