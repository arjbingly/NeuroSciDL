"""CLI Interface for training EEGViT model.

Example:
    python ViT2EEG_train_cli.py -c config.yaml fit --dev_run --data.annotation_file file_annotations.csv
"""
import torch
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from neuroscidl.callbacks import MLFlowSaveConfigCallback
from neuroscidl.eeg.eeg_dataset import EEGDataModule
from neuroscidl.eeg.eeg_model import EEGViT_pretrained
from neuroscidl.eeg.utils import filename_tagger, CalculateEEGDist
from pathlib import Path


class MyLightningCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the parser.
        model_prefix and dev_run are added to the parser.

        Args:
            parser (LightningArgumentParser): The argument parser to add arguments to.
        """
        parser.add_argument('--model_prefix', type=str, default='EEGViT')
        parser.add_argument('--dev_run', action='store_true', help='Run a development run')
        parser.add_argument('--tags', type=dict, default=None, help='Tags for MLFlow logger')
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
        tags = {}
        if self.config['fit']['tags'] is not None:
            tags.update(self.config['fit']['tags'])
        tags.update(filename_tagger(Path(self.config['fit']['data']['annotation_file']).name))
        self.config['fit']['trainer']['logger']['init_args']['tags'] = tags
        print(
            f"{Path(self.config['fit']['data']['annotation_file']).name}:{self.config['fit']['trainer']['logger']['init_args']['tags']}")

    def auto_transform_config(self, transform_index, transform_type='train'):
        """Auto set config_key and config_path for transforms."""
        if transform_index is None:
            if self.config['fit']['data'][f'{transform_type}_transform'].get('init_args') is not None:
                if self.config['fit']['data'][f'{transform_type}_transform']['init_args'].get('config_key') is not None:
                    if self.config['fit']['data'][f'{transform_type}_transform']['init_args']['config_key'] == 'auto':
                        print(f'Auto setting {transform_type}_transform config_key to {self.config['fit']['data']['annotation_file']}.')
                        self.config['fit']['data'][f'{transform_type}_transform']['init_args'][
                            'config_key'] = self.config['fit']['data']['annotation_file']
                    else:
                        if self.config['fit']['data'][f'{transform_type}_transform']['init_args']['config_key'] != self.config['fit']['data']['annotation_file']:
                            print(f'Warning: {transform_type}_transform config_key is not same as annotation_file.'
                                  f' It is recommended to set this to "auto".')
                if self.config['fit']['data'][f'{transform_type}_transform']['init_args'].get(
                    'config_path') is not None:
                    if self.config['fit']['data'][f'{transform_type}_transform']['init_args'][
                        'config_path'] == 'auto':
                        print(f'Auto setting {transform_type}_transform config_path to '
                              f'{self.config['fit']['data']['data_dir']}/noise_config.json')
                        self.config['fit']['data'][f'{transform_type}_transform']['init_args'][
                            'config_path'] = f"{self.config['fit']['data']['data_dir']}/noise_config.json"
                    else:
                        if self.config['fit']['data'][f'{transform_type}_transform']['init_args']['config_path'] != f"{self.config['fit']['data']['data_dir']}/noise_config.json'":
                            print(f'Warning: {transform_type}_transform config_path is not '
                                  f'{self.config['fit']['data']['data_dir']}/noise_config.json. '
                                  f'It is recommended to set this to "auto".')

        if self.config['fit']['data'][f'{transform_type}_transform'][transform_index].get('init_args') is not None:
            if self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args'].get('config_key') is not None:
                if self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args']['config_key'] == 'auto':
                    print(f'Auto setting {transform_type}_transform[{transform_index}] config_key to '
                          f'{self.config['fit']['data']['annotation_file']}.')
                    self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args'][
                        'config_key'] = self.config['fit']['data']['annotation_file']
                else:
                    if self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args']['config_key'] != self.config['fit']['data']['annotation_file']:
                        print(f'Warning: {transform_type}_transform[{transform_index}] config_key is not same as '
                              f'annotation_file. It is recommended to set this to "auto".')

            if self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args'].get(
                'config_path') is not None:
                if self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args'][
                    'config_path'] == 'auto':
                    print(f'Auto setting {transform_type}_transform[{transform_index}] config_path to '
                          f'{self.config['fit']['data']['data_dir']}/noise_config.json')
                    self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args'][
                        'config_path'] = f"{self.config['fit']['data']['data_dir']}/noise_config.json"
                else:
                    if self.config['fit']['data'][f'{transform_type}_transform'][transform_index]['init_args'][
                        'config_path'] != self.config['fit']['data']['annotation_file']:
                        print(f'Warning: {transform_type}_transform[{transform_index}] config_path is not '
                              f'{self.config['fit']['data']['data_dir']}/noise_config.json. '
                              f'It is recommended to set this to "auto".')

    def verify_transform_config_params(self):
        """Validate train_transform config_key and config_path."""
        if isinstance(self.config['fit']['data']['train_transform'], list):
            for transform_index, transform in enumerate(self.config['fit']['data']['train_transform']):
                self.auto_transform_config(transform_index, 'train')
        else:
            self.auto_transform_config(None, 'train')

    def validate_transform_config_file(self):
        """Checks if config for file exists in noise_config.json else create."""
        transform_configs = []
        if isinstance(self.config['fit']['data']['train_transform'], list):
            for transform in self.config['fit']['data']['train_transform']:
                t_config = {}
                if transform.get('init_args') is not None:
                    if transform['init_args'].get('config_key') is not None:
                        t_config['config_key'] = transform['init_args']['config_key']
                        if transform['init_args'].get('config_path') is not None:
                            t_config['config_path'] = transform['init_args']['config_path']
                            transform_configs.append(t_config)
        else:
            if self.config['fit']['data']['train_transform'].get('init_args') is not None:
                t_config = {}
                if self.config['fit']['data']['train_transform']['init_args'].get('config_key') is not None:
                    t_config['config_key'] = self.config['fit']['data']['train_transform']['init_args']['config_key']
                    if self.config['fit']['data']['train_transform']['init_args'].get('config_path') is not None:
                        t_config['config_path'] = self.config['fit']['data']['train_transform']['init_args']['config_path']
                        transform_configs.append(t_config)

        print('Checking noise config...')
        for t_config in transform_configs:
            calculator = CalculateEEGDist(self.config['fit']['data']['data_dir'])
            calculator(Path(t_config['config_path'])/t_config['config_key'])

    def auto_annotation_dir(self):
        """Auto set annotation_dir if set to 'auto'."""
        if self.config['fit']['data'].get('annotation_dir') == 'auto':
            self.config['fit']['data']['annotation_dir'] = Path(self.config['fit']['data']['data_dir'])/'annotations'
        else:
            print('Using given annotation_dir')

    def before_instantiate_classes(self) -> None:
        """Modify the configuration before instantiating classes."""
        # Auto annotation_dir
        self.auto_annotation_dir()

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
            self.verify_transform_config_params()
            # Checks if config for file exists in noise_config.json else create
            self.validate_transform_config_file()

def cli_main():
    """Main function to run the CLI."""
    torch.set_float32_matmul_precision('high')
    cli = MyLightningCli(EEGViT_pretrained,
                         EEGDataModule,
                         save_config_callback=MLFlowSaveConfigCallback,
                         parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == '__main__':
    cli_main()
