from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from neuroscidl.callbacks import MLFlowSaveConfigCallback
from neuroscidl.eeg.eeg_dataset import EEGDataModule
from neuroscidl.eeg.eeg_model import EEGViT_pretrained
from neuroscidl.eeg.utils import filename_tagger


class MyLightningCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument('--model_prefix', type=str, default='EEGViT')
        parser.add_argument('--dev_run', action='store_true', help='Run a development run')
        # parser.link_arguments('data.annotation_file', 'data.train_transform.init_args.config_key')

    def before_instantiate_classes(self) -> None:
        if self.config['fit'].get('dev_run'):
            print('Running a development run....')
            self.config['fit']['trainer']['max_epochs'] = 2
            self.config['fit']['trainer']['limit_train_batches'] = 10
            self.config['fit']['trainer']['limit_val_batches'] = 10

        ds_name = Path(self.config['fit']['data']['annotation_file']).name.replace("file_annotations_", "").split('.')[0]
        for i, cb in enumerate(self.config['fit']['trainer']['callbacks']):
            if 'ModelCheckpoint' in cb['class_path']:
                self.config['fit']['trainer']['callbacks'][i]['init_args']['filename'] =\
                    (f"{self.config['fit']['model_prefix']}_{ds_name}" +
                     '-{epoch:02d}-{' + f"{self.config['fit']['trainer']['callbacks'][i]['init_args']['monitor']}" +':.2f}')

        if self.config['fit']['trainer'].get('logger') is not None:
            if str(self.config['fit']['trainer']['logger']['class_path']) == 'lightning.pytorch.loggers.MLFlowLogger':
                self.config['fit']['trainer']['logger']['init_args']['tags'] = filename_tagger(Path(self.config['fit']['data']['annotation_file']).name)

        if self.config['fit']['data'].get('train_transform') is not None:
            if self.config['fit']['data']['train_transform'].get('init_args') is not None:
                if self.config['fit']['data']['train_transform']['init_args'].get('config_key') is not None:
                    if self.config['fit']['data']['train_transform']['init_args']['config_key'] != self.config['fit']['data']['annotation_file']:
                        print('Warning: train_transform config_key does not match annotation_file')


def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = MyLightningCli(EEGViT_pretrained,
                         EEGDataModule,
                         save_config_callback=MLFlowSaveConfigCallback,
                         parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == '__main__':
    cli_main()
