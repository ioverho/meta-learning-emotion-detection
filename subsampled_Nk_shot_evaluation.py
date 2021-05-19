import os
from copy import deepcopy
import pickle
import argparse
import re
from distutils.util import strtobool
from collections import defaultdict
import re

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from models.seqtransformer import SeqTransformer
from baseline.models.custombert import CustomBERT
from data.meta_dataset import meta_dataset, task_label_dict
from data.utils.tokenizer import manual_tokenizer, specials
from data.utils.data_loader_numpy import StratifiedLoader, StratifiedLoaderwClassesSubset
from data.utils.sampling import dataset_sampler
from utils.metrics import logging_metrics
from utils.timing import Timer
from utils.seed import set_seed


def eval(args):

    def _get_dataloader(datasubset, tokenizer, device, args, subset_classes=True):
        """
        Get specific dataloader.

        Args:
            datasubset ([type]): [description]
            tokenizer ([type]): [description]
            device ([type]): [description]
            args ([type]): [description]

        Returns:
            dataloader
        """

        if subset_classes:
            dataloader = StratifiedLoaderwClassesSubset(datasubset, k=args['k'],
                                                        max_classes=args['max_classes'],
                                                        max_batch_size=args['max_batch_size'],
                                                        tokenizer=tokenizer,
                                                        device=device,
                                                        shuffle=True,
                                                        verbose=False)
        else:
            dataloader = StratifiedLoader(datasubset, k=args['k'],
                                          max_batch_size=args['max_batch_size'],
                                          tokenizer=tokenizer,
                                          device=device,
                                          shuffle=True,
                                          verbose=False)

        return dataloader

    def _adapt_and_fit(support_labels_list, support_input_list, query_labels_list, query_input_list, loss_fn, model_init, args, mode):
        """
        Adapts the init model to a support set and computes loss on query set.

        Args:
            support_labels ([type]): [description]
            support_text ([type]): [description]
            query_labels ([type]): [description]
            query_text ([type]): [description]
            model_init ([type]): [description]
            args
            mode
        """

        #####################
        # Create model_task #
        #####################
        model_init.eval()

        model_task = deepcopy(model_init)
        model_task_optimizer = optim.SGD(model_task.parameters(),
                                        lr=args['inner_lr'])
        model_task.zero_grad()

        #######################
        # Generate prototypes #
        #######################

        with torch.no_grad():
            prototypes = 0.0
            for support_labels, support_input in zip(support_labels_list, support_input_list):
                if mode != "baseline":
                    y = model_init(support_input)
                else:
                    y = model_init.encode(support_input)

                labs = torch.sort(torch.unique(support_labels))[0]
                prototypes += torch.stack([torch.mean(y[support_labels == c], dim=0) for c in labs])

            prototypes = prototypes / len(support_labels_list)

            W_init = 2 * prototypes
            b_init = -torch.norm(prototypes, p=2, dim=1)**2

        W_task, b_task = W_init.detach(), b_init.detach()
        W_task.requires_grad, b_task.requires_grad = True, True

        #################
        # Adapt to data #
        #################
        for _ in range(args['n_inner']):
            for support_labels, support_input in zip(support_labels_list, support_input_list):
                if mode != "baseline":
                    y = model_task(support_input)
                else:
                    y = model_task.encode(support_input)

                logits = F.linear(y, W_task, b_task)

                inner_loss = loss_fn(logits, support_labels)

                W_task_grad, b_task_grad = torch.autograd.grad(inner_loss,
                                                                [W_task, b_task], retain_graph=True)

                inner_loss.backward()

                if args['clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(model_task.parameters(),
                                                    args['clip_val'])

                model_task_optimizer.step()

                W_task = W_task - args['output_lr'] * W_task_grad
                b_task = b_task - args['output_lr'] * b_task_grad

        #########################
        # Validate on query set #
        #########################
        logits_list, outer_loss_list = [], []
        for query_labels, query_input in zip(query_labels_list, query_input_list):
            with torch.no_grad():
                if mode != "baseline":
                    y = model_task(query_input)
                else:
                    y = model_task.encode(query_input)

                logits = F.linear(y, W_task, b_task)

                outer_loss = loss_fn(logits, query_labels)

                logits_list.append(logits)
                outer_loss_list.append(outer_loss)

        return logits_list, outer_loss_list

    #######################
    # Logging Directories #
    #######################
    log_dir = os.path.join(args['checkpoint_path'], args['version'])

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'evaluation'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoint'), exist_ok=True)
    #print(f"Saving models and logs to {log_dir}")

    checkpoint_save_path = os.path.join(log_dir, 'checkpoint')

    if args['mode'] != "baseline":
        with open(os.path.join("./", checkpoint_save_path, "hparams.pickle"), mode='rb+') as f:
            hparams = pickle.load(f)
    else:
        with open(os.path.join("./", args['checkpoint_path'], "hparams.pickle"), mode='rb+') as f:
            hparams = pickle.load(f)

    ##########################
    # Device, Logging, Timer #
    ##########################

    set_seed(args['seed'])

    timer = Timer()

    device = torch.device('cuda' if (torch.cuda.is_available() and args['gpu']) else 'cpu')

    # Build the tensorboard writer
    writer = SummaryWriter(os.path.join(log_dir, 'evaluation'))

    ###################
    # Load in dataset #
    ###################
    print("Data Prep")
    dataset = meta_dataset(include=args['include'], verbose=True)
    dataset.prep(text_tokenizer=manual_tokenizer)
    print("")

    ####################
    # Init models etc. #
    ####################
    if args['mode'] != "baseline":
        model_init = SeqTransformer(hparams)
        tokenizer = AutoTokenizer.from_pretrained(hparams['encoder_name'])
    else:
        model_init = CustomBERT(num_classes=task_label_dict[args['version']])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    tokenizer.add_special_tokens({'additional_special_tokens': specials()})
    model_init.encoder.model.resize_token_embeddings(len(tokenizer.vocab))

    for file in os.listdir(checkpoint_save_path):
        if 'best_model' in file:
            fp = os.path.join(checkpoint_save_path, file)
            with open(fp, mode='rb+') as f:
                print(f"Found pre-trained file at {fp}")
                if args['mode'] != "baseline":
                    model_init.load_state_dict(torch.load(f, map_location=device))

                    for name, param in model_init.encoder.model.named_parameters():
                        transformer_layer = re.search("(?:encoder/.layer/.)([0-9]+)", name)
                        if transformer_layer and (int(transformer_layer.group(1)) > args['nu']):
                            param.requires_grad = True
                        elif 'pooler' in name:
                            param.requires_grad = False
                        elif args['nu'] < 0:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    model_init.load_state_dict(torch.load(f, map_location=device)["bert_state_dict"])

    model_init = model_init.to(device)

    loss_fn = nn.CrossEntropyLoss()

    ##############
    # Evaluation #
    ##############

    results_dict = defaultdict(dict)

    for split in args['splits']:

        overall_loss, overall_acc, overall_f1 = [], [], []
        overall_loss_s, overall_acc_s, overall_f1_s = [], [], []

        ###################
        # Individual Task #
        ###################
        for task in dataset.lens.keys():
            datasubset = dataset.datasets[task][split]

            task_loss, task_acc, task_f1 = [], [], []
            task_loss_s, task_acc_s, task_f1_s = [], [], []
            for _ in range(args['n_eval_per_task']):
                dataloader = _get_dataloader(datasubset, tokenizer, device,
                                             args, subset_classes=args['subset_classes'])

                total_size = args['k'] * dataloader.n_classes
                n_sub_batches = total_size / args['max_batch_size']
                reg_k = int(args['k'] // n_sub_batches)
                left_over = args['k'] * dataloader.n_classes - \
                    int(n_sub_batches) * reg_k * dataloader.n_classes
                last_k = int(left_over / dataloader.n_classes)


                support_labels_list, support_input_list, query_labels_list, query_input_list = [], [], [], []

                dataloader.k = reg_k
                for _ in range(int(n_sub_batches)):

                    support_labels, support_text, query_labels, query_text = next(dataloader)

                    support_labels_list.append(support_labels)
                    support_input_list.append(support_text)
                    query_labels_list.append(query_labels)
                    query_input_list.append(query_text)

                if last_k > 0.0:
                    dataloader.k = last_k
                    support_labels, support_text, query_labels, query_text = next(dataloader)

                    support_labels_list.append(support_labels)
                    support_input_list.append(support_text)
                    query_labels_list.append(query_labels)
                    query_input_list.append(query_text)

                logits_list, loss_list = _adapt_and_fit(support_labels_list, support_input_list,
                                                        query_labels_list, query_input_list,
                                                        loss_fn, model_init, hparams, args['mode'])


                for logits, query_labels, loss in zip(logits_list, query_labels_list, loss_list):
                    mets = logging_metrics(logits.detach().cpu(), query_labels.detach().cpu())

                    task_loss.append(loss.detach().cpu().item())
                    task_acc.append(mets['acc'])
                    task_f1.append(mets['f1'])

                    task_loss_s.append(loss.detach().cpu().item() / np.log(dataloader.n_classes))
                    task_acc_s.append(mets['acc'] / (1/dataloader.n_classes))
                    task_f1_s.append(mets['f1'] / (1/dataloader.n_classes))

            overall_loss.append(np.mean(task_loss))
            overall_acc.append(np.mean(task_acc))
            overall_f1.append(np.mean(task_f1))

            overall_loss_s.append(np.mean(task_loss_s))
            overall_acc_s.append(np.mean(task_acc_s))
            overall_f1_s.append(np.mean(task_f1_s))

            print("{:} | Eval  | Split {:^8s} | Task {:^20s} | Loss {:5.2f} ({:4.2f}), Acc {:5.2f} ({:4.2f}), F1 {:5.2f} ({:4.2f}) | Mem {:5.2f} GB".format(
                timer.dt(), split, task,
                overall_loss_s[-1] if args['print_scaled'] else overall_loss[-1],
                np.std(task_loss_s) if args['print_scaled'] else np.std(task_loss),
                overall_acc_s[-1] if args['print_scaled'] else overall_acc[-1],
                np.std(task_acc_s) if args['print_scaled'] else np.std(task_acc),
                overall_f1_s[-1] if args['print_scaled'] else overall_f1[-1],
                np.std(task_f1_s) if args['print_scaled'] else np.std(task_f1),
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))

            writer.add_scalars(f'Loss/{split}',
                            {task: overall_loss[-1]}, 0)
            writer.add_scalars(f'Accuracy/{split}',
                            {task: overall_acc[-1]}, 0)
            writer.add_scalars(f'F1/{split}', {task: overall_f1[-1]}, 0)

            writer.add_scalars(f'LossScaled/{split}',
                                {task: overall_loss_s[-1]}, 0)
            writer.add_scalars(f'AccuracyScaled/{split}',
                                {task: overall_acc_s[-1]}, 0)
            writer.add_scalars(f'F1Scaled/{split}',
                            {task: overall_f1_s[-1]}, 0)

            writer.flush()

            results_dict[task][split] = {
                "loss": "{:.2f} ({:.2f})".format(overall_loss[-1], np.std(task_loss)),
                "acc": "{:.2f} ({:.2f})".format(overall_acc[-1], np.std(task_acc)),
                "f1": "{:.2f} ({:.2f})".format(overall_f1[-1], np.std(task_f1)),
                "loss_scaled": "{:.2f} ({:.2f})".format(overall_loss_s[-1], np.std(task_loss_s)),
                "acc_scaled": "{:.2f} ({:.2f})".format(overall_acc_s[-1], np.std(task_acc_s)),
                "f1_scaled": "{:.2f} ({:.2f})".format(overall_f1_s[-1], np.std(task_f1_s)),
            }

        #######################
        # All Tasks Aggregate #
        #######################
        overall_loss = np.mean(overall_loss)
        overall_acc = np.mean(overall_acc)
        overall_f1 = np.mean(overall_f1)

        overall_loss_s = np.mean(overall_loss_s)
        overall_acc_s = np.mean(overall_acc_s)
        overall_f1_s = np.mean(overall_f1_s)

        print("{:} | MACRO-AGG | Eval  | Split {:^8s} | Loss {:5.2f}, Acc {:5.2f}, F1 {:5.2f}\n".format(
            timer.dt(), split,
            overall_loss_s if args['print_scaled'] else overall_loss,
            overall_acc_s if args['print_scaled'] else overall_acc,
            overall_f1_s if args['print_scaled'] else overall_f1))

        writer.add_scalar(f'Loss/Macro{split}', overall_loss, 0)
        writer.add_scalar(f'Accuracy/Macro{split}', overall_acc, 0)
        writer.add_scalar(f'F1/Macro{split}', overall_f1, 0)

        writer.add_scalar(f'LossScaled/Macro{split}', overall_loss_s, 0)
        writer.add_scalar(f'AccuracyScaled/Macro{split}',
                            overall_acc_s, 0)
        writer.add_scalar(f'F1Scaled/Macro{split}', overall_f1_s, 0)

        writer.flush()

    with open(os.path.join(log_dir, 'evaluation', 'results.pickle'), 'wb+') as file:
        pickle.dump(results_dict, file)

# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--include', default=['crowdflower', 'go_emotions'], type=str, nargs='+',
                        help='Which datasets to include. Default is all.',
                        choices=['go_emotions', 'crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                        'emotion-cause', 'grounded_emotions', 'ssec', 'tales-emotion', 'tec'])

    parser.add_argument('--splits', default=['test', 'validation', 'train'], type=str, nargs='+',
                        help='Which dataset splits to include. Default is train, valid, test.',
                        choices=['test', 'validation', 'train'])

    # Data loader hyperparameters
    parser.add_argument('--k', default=4, type=int,
                        help='The number of examples per class to load. Default is 4')

    parser.add_argument('--max_classes', default=8, type=int,
                        help='Maximum number of classes of a task to sample, if available. Default is 8')

    parser.add_argument('--subset_classes', default=True,
                        help='Whether or not to subset the classes of a task. Default is True.')

    parser.add_argument('--max_batch_size', default=32, type=int,
                        help='Maximum the batchsize is allowed to reach. Default is 32')

    # Training hyperparameters
    parser.add_argument('--n_inner', default=5, type=int,
                        help='Number of outer epochs. Default is 5')

    parser.add_argument('--n_eval_per_task', type=int, default=3,
                        help='Number of support sets to evaluate on for a single task.')

    # Saving hyperparameters
    parser.add_argument('--checkpoint_path', default='./checkpoints/Baselines', type=str,
                        help='Path where to store the checkpoint. Default is ./checkpoints/ProtoMAML_Rebuild')

    parser.add_argument('--version', default='crowdflower', type=str,
                        help='Name of current run version. Default is debug')

    parser.add_argument('--mode', default='baseline', type=str,
                        help='Whether or not to evaluate baseline models.')

    # Other hyperparameters
    parser.add_argument('--nu', default=5, type=int,
                        help='Random seed.')

    parser.add_argument('--seed', default=610, type=int,
                        help='Random seed.')

    parser.add_argument('--gpu', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether to use a GPU.')

    parser.add_argument('--print_inner_loss', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether or not to print the inner losses. Default is False.')

    parser.add_argument('--print_scaled', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether or not to print metrics scaled to random. Default is True.')

    # Parse the arguments
    args = parser.parse_args()
    args = vars(parser.parse_args())

    # Print eval parameters
    print('-----Evaluation PARAMETERS-----')
    for k, v in args.items():
        print(f"{k}: {v}")
    print('-----------------------------\n')

    # Evaluate the model
    eval(args)
