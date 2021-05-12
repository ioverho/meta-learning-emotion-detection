# load in packages
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import torch.nn as nn
import numpy as np
import psutil

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset

# own imports
from models.transformer_clf import Transformer_CLF
from data.meta_dataset import MetaDataset


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, device, cutoff=100):
        self.samples = []

        for label in sorted(dataset.keys()):
            for i, point in enumerate(dataset[label]):
                tokenized_input = tokenizer(point['text'],
                                    return_tensors='pt',
                                    padding='max_length',
                                    truncation=True).to(device)

                self.samples.append((tokenized_input['input_ids'].squeeze(), tokenized_input['attention_mask'].squeeze(),
                          label))
            #if i > cutoff:
                #break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_dataloaders(dataset, tokenizer, device, batch_size=8, extract='go_emotions', shuffle=True, num_workers=0):
    data_splits = {}
    if extract == 'go_emotions':
        for split in dataset[extract].keys():
            data_split = CustomDataset(dataset[extract][split], tokenizer, device)
            data_split_loader = torch.utils.data.DataLoader(data_split, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
            data_splits[split] = data_split_loader
    else:
        for split in dataset.datasets[extract].keys():
            data_split = CustomDataset(dataset[extract][split], tokenizer, device)
            data_split_loader = torch.utils.data.DataLoader(data_split, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
            data_splits[split] = data_split_loader

    return data_splits


class CLFTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.args = args
        # Create model
        self.model = Transformer_CLF(args)
        # # Create loss module
        self.loss_module = nn.CrossEntropyLoss()


    def forward(self, text, attn_mask):
        return self.model(text, attn_mask)


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


    def encode(self, text, attn_mask=None):
        return self.model.encode(text, attn_mask)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the train data loader.
        text, attn_mask, labels = batch
        preds = self.model(text, attn_mask)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # calculate the memory usage
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

        self.log('mem_usage', mem, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        text, attn_mask, labels = batch

        preds = self.model(text, attn_mask).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)


    def test_step(self, batch, batch_idx):
        text, attn_mask, labels = batch
        preds = self.model(text, attn_mask).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards


def main(args):
    """
    Function for handling the arguments and starting the experiment.
    Inputs:
        args - Namespace object from the argument parser
    """

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    print('Encoder name: {}'.format(args.encoder_name))
    print('Hidden dimensions: {}'.format(args.hidden_dims))
    print('Include: {}'.format(args.include))
    print('Num classes: {}'.format(args.num_classes))
    print('Activation function: {}'.format(args.act_fn))
    print('Max epochs: {}'.format(args.max_epochs))
    print('Nu: {}'.format(args.nu))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Checkpoint path: {}'.format(args.checkpoint_path))
    print('Version: {}'.format(args.version))
    print('Seed: {}'.format(args.seed))
    print('Show progress bar: {}'.format(args.progress_bar))
    print('PyTorch device: {}'.format(device))
    print('-----------------------------')

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    # load the dataset
    print("Extracting datasets..")
    # ToDo: process data and make sure it uses same amount of training data as protomaml
    dataset = MetaDataset(include=args.include)
    tokenizer_kwargs = {'return_tensors':'pt', 'padding':'max_length', 'truncation':True}
    dataset.prep(tokenizer)
    print("Datasets extracted")

    # create dataloaders for the dataset
    print("Creating dataloaders..")
    data_loaders = extract_dataloaders(dataset, tokenizer, device, args.batch_size, extract=args.include[0])
    train_loader = data_loaders['train']
    validation_loader = data_loaders['validation']
    test_loader = data_loaders['test']
    print("Dataloaders created")

    # create the trainer object
    print('Creating trainer..')
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path, save_weights_only=True, mode="max", monitor="val_acc")
    trainer = pl.Trainer(default_root_dir=os.path.join(args.checkpoint_path, args.version),
                         checkpoint_callback=checkpoint_callback,
                         gpus=1 if str(device)=="cuda" else 0,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0
                         )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    print('Trainer created')

    # seed for reproducability
    pl.seed_everything(args.seed)

    # train the model
    print('Starting training..')
    model = CLFTrainer(args)
    trainer.fit(model, train_loader, validation_loader)
    print('Training finished')

    print('Starting testing..')
    model = CLFTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    print('Testing finished')


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--encoder_name', default='bert-base-uncased', type=str,
                        help='What encoder model to use. Default is bert-base-uncased',
                        choices=['bert-base-uncased'])
    parser.add_argument('--hidden_dims', default=[256, 128], type=float, nargs='*',
                        help='Hidden dimensions for the model. Default is [256, 128]')
    parser.add_argument('--include', default=['go_emotions'], type=str, nargs='*',
                        help='Which datasets to include. Default is [go_emotions]',
                        choices=['go_emotions', 'crowdflower'])
    parser.add_argument('--num_classes', default=27, type=int,
                        help='Number of classes of the dataset. Default is 27')
    parser.add_argument('--act_fn', default='Tanh', type=str,
                        help='Which activation function to use in the model. Default is Tanh',
                        choices=['Tanh'])

    # training hyperparameters
    parser.add_argument('--max_epochs', default=4, type=int,
                        help='Maximum number of epochs to train for. Default is 4')

    # optimizer hyperparameters
    parser.add_argument('--nu', default=-1, type=int,
                        help='Nu to determine the amount of parameters optimized. Default is -1 (all parameters)')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Learning rate to use for the optimizer. Default is 5e-5')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size. Default is 16')

    # saving hyperparameters
    parser.add_argument('--checkpoint_path', default='./checkpoints/baselines', type=str,
                        help='Path to where the model checkpoint is stored. Default is ./checkpoints/baselines')
    parser.add_argument('--version', default='go_emotions_test', type=str,
                        help='Name of the model version. Default is go_emotions_test')

    # other hyperparameters
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    # parse the arguments
    args = parser.parse_args()

    # train the model
    main(args)
