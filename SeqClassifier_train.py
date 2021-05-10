import os
import argparse
from distutils.util import strtobool
import pickle

from memory_profiler import profile
import psutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from models.seqclassifier import SeqClassifer
from data.utils.data_loader_numpy import StratifiedLoader
from data.unified_emotion_numpy import unified_emotion
from data.utils.tokenizer import manual_tokenizer, specials
from utils.metrics import logging_metrics
from utils.timing import Timer

def data_examples(dataset, tokenizer):
        print('\nExample data')
        for task in dataset.lens.keys():
            data_subset = dataset.datasets[task]['train']
            dataloader = StratifiedLoader(data_subset,
                                          device='cpu',
                                          k=1)
            support_labels, support_text, _, _ = next(dataloader)

            print(task)

            label_map = {v: k for k, v in dataset.label_map[task].items()}
            tokenized_texts = list(
                map(tokenizer.decode, tokenizer(list(support_text))['input_ids']))
            for txt, label in zip(tokenized_texts, support_labels):
                print(label_map[label], txt)
            print()

@profile
def train(config):

    with profiler.profile(profile_memory=True) as prof, torch.autograd.set_detect_anomaly(True):
        # Set to debug in case of various weird tests
        if (not config['gpu']) and (not config['debug']):
            config['debug'] = True
            print(f"Setting debug mode to {config['debug']}.")
        if config['debug']:
            config['version'] = 'debug'

        ## Logging Directories
        log_dir = os.path.join(config['checkpoint_path'], config['version'])

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoint'), exist_ok=True)
        print(f"Saving models and logs to {log_dir}")

        with open(os.path.join(log_dir, 'checkpoint', 'hparams.pickle'), 'wb') as file:
            pickle.dump(config, file)

        ## Initialization
        # Device
        device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu']) else 'cpu')

        # Build the tensorboard writer
        writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

        # Load in the data
        dataset = unified_emotion(file_path="./data/datasets/unified-dataset.jsonl",
                                  include=config['include'], verbose=False)
        dataset.prep(text_tokenizer=manual_tokenizer)

        n_classes = dataset.info[0]['classes']
        config['n_classes'] = n_classes

        loss_fn= nn.CrossEntropyLoss()

        # Initialization of model
        model = SeqClassifer(config).to(device)

        # Huggingface tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['encoder_name'])

        tokenizer.add_special_tokens({'additional_special_tokens': specials()})
        model.encoder.model.resize_token_embeddings(len(tokenizer.vocab))

        # Meta optimizers
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
        lr_schedule = get_constant_schedule_with_warmup(optimizer, config['warmup_steps'])

        # Check the dataloader
        data_examples(dataset, tokenizer)

        # Set-up the timer
        timer = Timer()

        data_subset = dataset[config['include'][0]]['train']
        data_loader = StratifiedLoader(data_subset, k=16,
                                       shuffle=True, max_batch_size=config['max_batch_size'],
                                       tokenizer=tokenizer, device=device)

        for episode in range(1, config['max_episodes']+1):

            ############
            # Training #
            ############

            batch = next(data_loader)
            support_labels, support_text, _, _ = batch

            logits = model(support_text)
            loss = loss_fn(logits, support_labels)
            loss.backward()

            optimizer.step()
            lr_schedule.step()

            optimizer.zero_grad()

            with torch.no_grad():
                mets = logging_metrics(logits.detach().cpu(), support_labels.detach().cpu())

                loss, acc, f1 = loss.detach().item(), mets['acc'], mets['f1']

                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

                print("{:} | Train | Episode {:>04d} | Loss {:.4f}, Acc {:.4f}, F1 {:.4f} | Memory: {:5.2f}Gb"
                    .format(timer.dt(),
                            episode,
                            loss,
                            acc,
                            f1,
                            mem))

    #print(prof.key_averages().table(sort_by="cpu_memory_usage"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Dataset Initialization Hyperparameters
    parser.add_argument('--include', type=str, nargs=1, default=['crowdflower'],  # ['crowdflower', 'dailydialog', 'electoraltweets', 'emoint', 'emotion-cause', 'grounded_emotions', 'go_emotions', 'ssec', 'tec'],
                        choices=['crowdflower', 'dailydialog', 'electoraltweets', 'emoint', 'emotion-cause', 'grounded_emotions', 'go_emotions', 'ssec', 'tec'],
                        help='Dataset to include.')

    parser.add_argument('--max_batch_size', type=int, default=64,
                        help='Batch size during adaptation to support set.')

    ## Model Initialization Hyperparameters
    # Encoder
    parser.add_argument('--encoder_name', type=str, default='bert-base-cased',
                        help='Pretrained encoder model matching import from Hugginface, e.g. "bert-base-uncased", "vinai/bertweet-base".')

    parser.add_argument('--nu', type=int, default=5,
                        help='Max layer to keep frozen. 11 keeps enitre model frozen, -1 entirely trainable.')

    # Classifier
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden dimensions of the MLP. Pass a space separated list, e.g. "--hidden_dims 256 128".')

    parser.add_argument('--act_fn', type=str, default='Tanh',
                        help='Which activation to use. Currently either Tanh or ReLU.')

    ## Training Hyperparameters
    parser.add_argument('--max_episodes', type=int, default=10000,
                        help='Maximum number of episodes to take.')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the model update.')

    parser.add_argument('--warmup_steps', type=float, default=100,
                        help='Learning warm-up steps for the shared model update. Uses linear schedule to constant.')

    parser.add_argument('--clip_val', type=float, default=5,
                        help='Max norm of gradients to avoid exploding gradients.')

    ## MISC
    # Versioning, logging
    parser.add_argument('--version', type=str, default='supervised_numpy_training',
                        help='Construct model save name using versioning.')

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/Supervised",
                        help='Directory to save models to.')

    # Debugging
    parser.add_argument('--debug', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether to run in debug mode.')
    parser.add_argument('--gpu', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether to train on GPU (if available) or CPU.')

    config = vars(parser.parse_args())

    train(config)
