import os
import argparse
from collections import defaultdict
from distutils.util import strtobool
import pickle
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from models.protomaml_seqtransformer import ProtoMAMLSeqTransformer
from data.utils.data_loader import StratifiedLoader, AdaptiveNKShotLoader
from data.unified_emotion import unified_emotion
from data.utils.sampling import dataset_sampler
from data.utils.tokenizer import manual_tokenizer
from utils.metrics import logging_metrics
from utils.timing import Timer

def meta_evaluate(model, dataset, tokenizer, device, config, timer):
    """
    Check model performance on all datasets.
    DO NOT CALL with torch.no_grad(). THIS IS HANDLED INSIDE.

    Args:
        model: model currently being trained
        dataset: current dataset
        tokenizer (AutoTokenizer): Huggingface's tokenizer to match the model
        config (dict): training config dictionary
        k (int, optional): size of the k-shot. Defaults to 16.

    Returns:
        dict: dictionary with metrics per task

    """

    model.eval()

    task_vals = defaultdict(dict)

    for task in dataset.lens.keys():

        task_loss, task_acc, task_f1 = [], [], []
        for i in range(config['n_eval_per_task']):

            sample_loss, sample_acc, sample_f1 = [], [], []

            datasubset = dataset.datasets[task]['test']
            dataloader = AdaptiveNKShotLoader(dataset=datasubset,
                                              device=device,
                                              max_support_size=config['max_support_size'],
                                              tokenizer=tokenizer)

            # Inner loop
            # Support set
            support_labels, support_text, query_labels, query_text = next(dataloader)

            #model.train()
            model.adapt(support_labels, support_text, task_name=task)

            with torch.no_grad():
                for ii in range(config['n_eval_per_support']):

                    logits = model(query_text)
                    loss = model.lossfn(logits, query_labels)

                    mets = logging_metrics(
                        logits.detach().cpu(), query_labels.detach().cpu())

                    sample_loss.append(loss.item())
                    sample_acc.append(mets['acc'] * 100)
                    sample_f1.append(mets['f1'] * 100)

            task_loss.append(np.mean(sample_loss))
            task_acc.append(np.mean(sample_acc))
            task_f1.append(np.mean(sample_f1))
            #print('Task {:}: {:}/{:} | Loss {:.4E}, Acc {:5.2f}, F1 {:5.2f}'.format(task, i+1, config['n_eval_per_task'], \
            #    task_loss[-1], task_acc[-1], task_f1[-1]))

        print(u"{:} | Eval | Task {:} | Loss {:.2E} \u00B1 {:.2E}, Acc {:5.2f} \u00B1 {:4.2f}, F1 {:5.2f} \u00B1 {:4.2f}".format(timer.dt(),\
            task, np.mean(task_loss), np.std(task_loss), np.mean(task_acc), np.std(task_acc), np.mean(task_f1), np.std(task_f1)))

        task_vals['loss'][task] = np.mean(task_loss)
        task_vals['acc'][task] = np.mean(task_acc)
        task_vals['f1'][task] = np.mean(task_f1)

    return task_vals

def train(config):

    # Set to debug in case of various weird tests
    if (not config['gpu']) and (not config['debug']):
        config['debug'] = True
        print(f"Setting debug mode to {config['debug']}.")

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
    dataset = unified_emotion("./data/datasets/unified-dataset.jsonl",
                              include=config['include'])
    dataset.prep(text_tokenizer=manual_tokenizer)

    # Initialization of model
    config['lossfn'] = nn.CrossEntropyLoss
    model = ProtoMAMLSeqTransformer(config).to(device)
    print(f"Model loaded succesfully on device: {model.device()}")

    # Huggingface tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['encoder_name'])

    tokenizer.add_special_tokens({'additional_special_tokens': ["HTTPURL", "@USER"]})
    model.model_shared.encoder.model.resize_token_embeddings(len(tokenizer.vocab))

    # Meta optimizers
    shared_optimizer = optim.SGD(model.model_shared.parameters(), lr=config['meta_lr'])
    shared_lr_schedule = get_constant_schedule_with_warmup(shared_optimizer, config['warmup_steps'])

    # Check the dataloader
    print('\nExample data')
    for task in dataset.lens.keys():
        datasubset = dataset.datasets[task]['train']
        dataloader = StratifiedLoader(dataset=datasubset,
                                      device=device,
                                      k=1)
        support_labels, support_text, _, _ = next(dataloader)

        print(task)

        label_map = {v: k for k, v in dataset.label_map[task].items()}
        tokenized_texts = list(
            map(tokenizer.decode, tokenizer(support_text)['input_ids']))
        for txt, label in zip(tokenized_texts, support_labels):
            print(label_map[label], txt)
        print()

    # Set-up the timer
    timer = Timer()

    # Meta-evaluate prior to training for decent baseline
    if (not config['debug']):
        meta_eval = meta_evaluate(model, dataset, tokenizer, device, config, timer)

        macro_f1 = np.mean(list(meta_eval['f1'].values()))

        writer.add_scalars('Loss/MetaEval', meta_eval['loss'], 0)
        writer.add_scalars('Accuracy/MetaEval', meta_eval['acc'], 0)
        writer.add_scalars('F1/MetaEval', meta_eval['f1'], 0)
        writer.add_scalar('MacroF1/MetaEval', 0)

        best_macro_f1 = macro_f1
    else:
        best_macro_f1 = 0.0

    curr_patience = config['patience']

    if config['overfit_on_single_query']:
        assert len(config['include']) == 1, "Too many datasets for overfit test."
        dataloader = StratifiedLoader(dataset=dataset.datasets[list(dataset.lens.keys())[0]]['test'],
                                      device=device, k=16, tokenizer=tokenizer)
        overfit_labels, overfit_text, _, _ = next(dataloader)

    for episode in range(1, config['max_episodes']+1):
        ############
        # Training #
        ############
        for ii in range(config['n_outer']):

            source_name = dataset_sampler(dataset, sampling_method='sqrt')
            datasubset = dataset.datasets[task]['train']
            dataloader = AdaptiveNKShotLoader(dataset=datasubset,
                                              device=device,
                                              tokenizer=tokenizer,
                                              max_support_size=config['max_support_size'],
                                              temp_map=True
                                              )

            # Inner loop
            # Support set
            print(source_name)
            batch = next(dataloader)
            support_labels, support_input, query_labels, query_input  = batch

            if config['overfit_on_single_query']:
                query_labels, query_input = overfit_labels, overfit_text

            model.train()
            model.adapt(support_labels, support_input,
                        task_name=source_name, verbose=config['debug'])

            # Outer loop
            # Query set
            model.eval()
            logits = model(query_input)
            loss = model.lossfn(logits, query_labels)

            model.backward(loss)

            # Logging
            n_classes = len(dataset.label_map[source_name].keys())
            with torch.no_grad():
                mets = logging_metrics(logits.detach().cpu(), query_labels.detach().cpu())
            print("{:} | Train | Episode {} | Task {}/{}: {:<20s}, N={} | Loss {:.4E}, Acc {:5.2f}, F1 {:5.2f}"\
                .format(timer.dt(), episode, ii+1, config['n_outer'], source_name, n_classes,
                        loss.detach().item(), mets['acc']*100, mets['f1']*100))

            writer.add_scalars('Loss/Train', {source_name: loss.detach().item()}, episode)
            writer.add_scalars('Accuracy/Train', {source_name: mets['acc'] * 100}, episode)
            writer.add_scalars('F1/Train', {source_name: mets['f1'] * 100}, episode)

        shared_optimizer.step()
        shared_lr_schedule.step()
        shared_optimizer.zero_grad()

        ##############
        # Evaluation #
        ##############
        if (episode % config['eval_every_n']) == 0 and (not config['debug']):

            meta_eval = meta_evaluate(model, dataset, tokenizer, device, config, timer)

            macro_f1 = np.mean(list(meta_eval['f1'].values()))

            writer.add_scalars('Loss/MetaEval', meta_eval['loss'], episode)
            writer.add_scalars('Accuracy/MetaEval', meta_eval['acc'], episode)
            writer.add_scalars('F1/MetaEval', meta_eval['f1'], episode)
            writer.add_scalar('MacroF1/MetaEval', episode+1)

            if macro_f1 > best_macro_f1:
                save_name = "episode-{:}_macrof1-{:5.2f}".format(episode, macro_f1)
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'checkpoint', save_name))

                print(f"Saving model as {save_name}")
                best_macro_f1 = macro_f1
                curr_patience = config['patience']

            else:
                print(f"Model did not improve with macrof1={macro_f1}")
                if episode > config['min_episodes']:
                    curr_patience -= 1

            print('')

            if curr_patience < 0:
                print("Stopping early.")
                break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Dataset Initialization Hyperparameters
    parser.add_argument('--include', type=str, nargs='+', default=['crowdflower', 'dailydialog', 'electoraltweets', 'emoint', 'emotion-cause', 'grounded_emotions', 'ssec', 'tec'],
                        help='Datasets to include.')

    parser.add_argument('--max_support_size', type=int, default=8,
                        help='Batch size during adaptation to support set.')

    ## Model Initialization Hyperparameters
    # Encoder
    parser.add_argument('--encoder_name', type=str, default='bert-base-uncased',
                        help='Pretrained encoder model matching import from Hugginface, e.g. "bert-base-uncased", "vinai/bertweet-base".')

    parser.add_argument('--nu', type=int, default=11,
                        help='Max layer to keep frozen. 11 keeps enitre model frozen, -1 entirely trainable.')

    # Classifier
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden dimensions of the MLP. Pass a space separated list, e.g. "--hidden_dims 256 128".')

    parser.add_argument('--act_fn', type=str, default='Tanh',
                        help='Which activation to use. Currently either Tanh or ReLU.')

    ## Meta-training Hyperparameters
    # MAML

    parser.add_argument('--n_inner', type=int, default=7,
                        help='Number of inner loop (MAML) steps to take.')

    parser.add_argument('--n_outer', type=int, default=1,
                        help='Number of outer loop (MAML) steps to take. Samples a new task per steps. Values greater than 1 essentially mean accumulated gradients.')

    parser.add_argument('--max_episodes', type=int, default=10000,
                        help='Maximum number of episodes to take.')

    parser.add_argument('--min_episodes', type=int, default=7500,
                        help='Minimum number of episodes to take. Starts checking early stopping after this is reached.')

    parser.add_argument('--patience', type=int, default=2,
                        help='Number of evaluations without improvement before stopping training.')

    # Optimizer
    parser.add_argument('--meta_lr', type=float, default=1e-5,
                        help='Learning rate for the shared model update.')

    parser.add_argument('--inner_lr', type=float, default=1e-3,
                        help='Learning rate for the task-specific model update.')

    parser.add_argument('--output_lr', type=float, default=1e-3,
                        help='Learning rate for the softmax classification layer update.')

    parser.add_argument('--warmup_steps', type=float, default=100,
                        help='Learning warm-up steps for the shared model update. Uses linear schedule to constant.')

    parser.add_argument('--clip_val', type=float, default=5,
                        help='Max norm of gradients to avoid exploding gradients.')

    ## Meta-eval Hyperparameters

    parser.add_argument('--n_eval_per_task', type=int, default=3,
                        help='Number of support sets to try for a single task.')

    parser.add_argument('--n_eval_per_support', type=int, default=1,
                        help='Number of different batches to evaluate on per support set.')

    parser.add_argument('--eval_every_n', type=int, default=250,
                        help='Number of different batches to evaluate on per support set.')

    ## MISC
    # Versioning, logging
    parser.add_argument('--version', type=str, default='debug',
                        help='Construct model save name using versioning.')

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/ProtoMAML",
                        help='Directory to save models to.')

    # Debugging
    parser.add_argument('--overfit_on_single_query', default=False, type=lambda x: bool(strtobool(x)),
                        help='As a check for learning, overfits model on a single query set.')
    parser.add_argument('--debug', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether to run in debug mode.')
    parser.add_argument('--gpu', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether to train on GPU (if available) or CPU.')

    config = vars(parser.parse_args())

    train(config)
