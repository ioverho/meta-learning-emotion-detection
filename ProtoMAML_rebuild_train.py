import os
from copy import deepcopy
import pickle
import argparse
import re
from distutils.util import strtobool

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from models.seqtransformer import SeqTransformer
from data.meta_dataset import meta_dataset
from data.utils.tokenizer import manual_tokenizer, specials
from data.utils.data_loader_numpy import StratifiedLoader, StratifiedLoaderwClassesSubset
from data.utils.sampling import dataset_sampler
from utils.metrics import logging_metrics
from utils.timing import Timer
from utils.seed import set_seed


def train(args):

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

    def _adapt_and_fit(support_labels, support_input, query_labels, query_input, loss_fn, model_init, args, mode="train"):
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
        if (not args['dropout']) and mode == "train":
            for module in model_init.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
                else:
                    module.train()
        elif mode != "train":
            model_init.eval()
        else:
            model_init.train()

        model_task = deepcopy(model_init)
        model_task_optimizer = optim.SGD(model_task.parameters(),
                                         lr=args['inner_lr'])
        model_task.zero_grad()

        #######################
        # Generate prototypes #
        #######################

        y = model_init(support_input)

        labs = torch.sort(torch.unique(support_labels))[0]
        prototypes = torch.stack([torch.mean(y[support_labels == c], dim=0) for c in labs])

        W_init = 2 * prototypes
        b_init = -torch.norm(prototypes, p=2, dim=1)**2

        W_task, b_task = W_init.detach(), b_init.detach()
        W_task.requires_grad, b_task.requires_grad = True, True

        #################
        # Adapt to data #
        #################
        for _ in range(args['n_inner']):

            y = model_task(support_input)
            logits = F.linear(y, W_task, b_task)

            inner_loss = loss_fn(logits, support_labels)

            W_task_grad, b_task_grad = torch.autograd.grad(inner_loss,\
                [W_task, b_task], retain_graph=True)

            inner_loss.backward()

            if args['clip_val'] > 0:
                torch.nn.utils.clip_grad_norm_(model_task.parameters(),
                                               args['clip_val'])

            model_task_optimizer.step()

            W_task = W_task - args['output_lr'] * W_task_grad
            b_task = b_task - args['output_lr'] * b_task_grad

            if args['print_inner_loss']:
                print(f"\tInner Loss: {inner_loss.detach().cpu().item()}")

        #########################
        # Validate on query set #
        #########################
        if mode == "train":
            for module in model_task.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

            W_task = W_init + (W_task - W_init).detach()
            b_task = b_init + (b_task - b_init).detach()

        y = model_task(query_input)
        logits = F.linear(y, W_task, b_task)

        outer_loss = loss_fn(logits, query_labels)

        if mode == "train":
            model_task_params = [param for param in model_task.parameters() if param.requires_grad]
            model_task_grads = torch.autograd.grad(outer_loss, model_task_params,
                                                   retain_graph=True)

            model_init_params = [param for param in model_init.parameters() if param.requires_grad]
            model_init_grads = torch.autograd.grad(outer_loss, model_init_params,
                                                   retain_graph=False)

            model_init_grads = model_init_grads + model_task_grads

            for param, grad in zip(model_init_params, model_init_grads):
                if param.grad != None:
                    param.grad += grad.detach()
                else:
                    param.grad = grad.detach()
        else:
            del model_task, W_task, b_task, W_task_grad, b_task_grad, prototypes, W_init, b_init

        if outer_loss.detach().cpu().item() > 10:
            print(outer_loss.detach().cpu().item(),
                  inner_loss.detach().cpu().item())

        return logits.detach(), outer_loss.detach()

    #######################
    # Logging Directories #
    #######################
    log_dir = os.path.join(args['checkpoint_path'], args['version'])

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoint'), exist_ok=True)
    #print(f"Saving models and logs to {log_dir}")

    checkpoint_save_path = os.path.join(log_dir, 'checkpoint')

    with open(os.path.join(log_dir, 'checkpoint', 'hparams.pickle'), 'wb') as file:
        pickle.dump(args, file)

    ##########################
    # Device, Logging, Timer #
    ##########################

    set_seed(args['seed'])

    timer = Timer()

    device = torch.device('cuda' if (torch.cuda.is_available() and args['gpu']) else 'cpu')

    # Build the tensorboard writer
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

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
    model_init = SeqTransformer(args)
    tokenizer = AutoTokenizer.from_pretrained(args['encoder_name'])

    tokenizer.add_special_tokens({'additional_special_tokens': specials()})
    model_init.encoder.model.resize_token_embeddings(len(tokenizer.vocab))

    meta_optimizer = optim.Adam(model_init.parameters(), lr=args['meta_lr'])
    meta_scheduler = get_constant_schedule_with_warmup(meta_optimizer, args['warmup_steps'])
    reduceOnPlateau = optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, mode='max',
                                                           factor=args['lr_reduce_factor'],
                                                           patience=args['patience'],
                                                           verbose=True)


    model_init = model_init.to(device)

    loss_fn = nn.CrossEntropyLoss()

    #################
    # Training loop #
    #################

    best_overall_acc_s = 0.0
    curr_patience = args['patience']

    for episode in range(1, args['max_episodes']+1):

        outer_loss_agg, acc_agg, f1_agg = 0.0, 0.0, 0.0
        outer_loss_s_agg, acc_s_agg, f1_s_agg = 0.0, 0.0, 0.0

        for ii in range(1, args['n_outer']+1):
            #################
            # Sample a task #
            #################
            task = dataset_sampler(dataset, sampling_method='sqrt')

            datasubset = dataset.datasets[task]['train']

            dataloader = _get_dataloader(datasubset, tokenizer, device,
                                         args, subset_classes=args['subset_classes'])

            support_labels, support_input, query_labels, query_input = next(dataloader)

            logits, outer_loss = _adapt_and_fit(support_labels, support_input,
                                                query_labels, query_input,
                                                loss_fn, model_init, args,
                                                mode="train")

            ######################
            # Inner Loop Logging #
            ######################
            with torch.no_grad():
                mets = logging_metrics(logits.detach().cpu(), query_labels.detach().cpu())
                outer_loss_ = outer_loss.detach().cpu().item()
                acc = mets['acc']
                f1 = mets['f1']

                outer_loss_s = outer_loss_/ np.log(dataloader.n_classes)
                acc_s = acc / (1/dataloader.n_classes)
                f1_s = f1/(1/dataloader.n_classes)

                outer_loss_agg += outer_loss_ / args['n_outer']
                acc_agg += acc / args['n_outer']
                f1_agg += f1 / args['n_outer']

                outer_loss_s_agg += outer_loss_s / args['n_outer']
                acc_s_agg += acc_s / args['n_outer']
                f1_s_agg += f1_s / args['n_outer']

            print("{:} | Train | Episode {:04}.{:02} | Task {:^20s}, N={:} | Loss {:5.2f}, Acc {:5.2f}, F1 {:5.2f} | Mem {:5.2f} GB".format(
                timer.dt(), episode, ii, task, dataloader.n_classes,
                outer_loss_s if args['print_scaled'] else outer_loss_,
                acc_s if args['print_scaled'] else acc,
                f1_s if args['print_scaled'] else f1,
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))

            writer.add_scalars('Loss/Train', {task: outer_loss_}, episode)
            writer.add_scalars('Accuracy/Train', {task: acc}, episode)
            writer.add_scalars('F1/Train', {task: f1}, episode)

            writer.add_scalars('LossScaled/Train', {task: outer_loss_s}, episode)
            writer.add_scalars('AccuracyScaled/Train', {task: acc_s}, episode)
            writer.add_scalars('F1Scaled/Train', {task: f1_s}, episode)

            writer.flush()

        ############################
        # Init Model Backward Pass #
        ############################
        model_init_params = [param for param in model_init.parameters() if param.requires_grad]
        #for param in model_init_params:
        #    param.grad = param.grad #/ args['n_outer']

        if args['clip_val'] > 0:
            torch.nn.utils.clip_grad_norm_(model_init_params, args['clip_val'])

        meta_optimizer.step()
        meta_scheduler.step()
        meta_optimizer.zero_grad()

        #####################
        # Aggregate Logging #
        #####################
        print("{:} | MACRO-AGG | Train | Episode {:04} | Loss {:5.2f}, Acc {:5.2f}, F1 {:5.2f}\n".format(
            timer.dt(), episode,
            outer_loss_s_agg if args['print_scaled'] else outer_loss_agg,
            acc_s_agg if args['print_scaled'] else acc_agg,
            f1_s_agg if args['print_scaled'] else f1_agg))

        writer.add_scalar('Loss/MacroTrain', outer_loss_agg, episode)
        writer.add_scalar('Accuracy/MacroTrain', acc_agg, episode)
        writer.add_scalar('F1/MacroTrain', f1_agg, episode)

        writer.add_scalar('LossScaled/MacroTrain', outer_loss_s_agg, episode)
        writer.add_scalar('AccuracyScaled/MacroTrain', acc_s_agg, episode)
        writer.add_scalar('F1Scaled/MacroTrain', f1_s_agg, episode)

        writer.flush()

        ##############
        # Evaluation #
        ##############
        if (episode % args['eval_every_n']) == 0:

            overall_loss, overall_acc, overall_f1 = [], [], []
            overall_loss_s, overall_acc_s, overall_f1_s = [], [], []
            ###################
            # Individual Task #
            ###################
            for task in dataset.lens.keys():
                datasubset = dataset.datasets[task]['validation']

                task_loss, task_acc, task_f1 = [], [], []
                task_loss_s, task_acc_s, task_f1_s = [], [], []
                for _ in range(args['n_eval_per_task']):

                    dataloader = _get_dataloader(datasubset, tokenizer, device,
                                                 args, subset_classes=args['subset_classes'])
                    support_labels, support_input, query_labels, query_input = next(dataloader)

                    logits, loss = _adapt_and_fit(support_labels, support_input,
                                                  query_labels, query_input,
                                                  loss_fn, model_init, args,
                                                  mode="eval")

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

                print("{:} | Eval  | Episode {:04} | Task {:^20s} | Loss {:5.2f}, Acc {:5.2f}, F1 {:5.2f} | Mem {:5.2f} GB".format(
                    timer.dt(), episode, task,
                    overall_loss_s[-1] if args['print_scaled'] else overall_loss[-1],
                    overall_acc_s[-1] if args['print_scaled'] else overall_acc[-1],
                    overall_f1_s[-1] if args['print_scaled'] else overall_f1[-1],
                    psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))

                writer.add_scalars('Loss/Eval', {task: overall_loss[-1]}, episode)
                writer.add_scalars('Accuracy/Eval', {task: overall_acc[-1]}, episode)
                writer.add_scalars('F1/Eval', {task: overall_f1[-1]}, episode)

                writer.add_scalars('LossScaled/Eval', {task: overall_loss_s[-1]}, episode)
                writer.add_scalars('AccuracyScaled/Eval', {task: overall_acc_s[-1]}, episode)
                writer.add_scalars('F1Scaled/Eval', {task: overall_f1_s[-1]}, episode)

                writer.flush()

            #######################
            # All Tasks Aggregate #
            #######################
            overall_loss = np.mean(overall_loss)
            overall_acc = np.mean(overall_acc)
            overall_f1 = np.mean(overall_f1)

            overall_loss_s = np.mean(overall_loss_s)
            overall_acc_s = np.mean(overall_acc_s)
            overall_f1_s = np.mean(overall_f1_s)

            print("{:} | MACRO-AGG | Eval  | Episode {:04} | Loss {:5.2f}, Acc {:5.2f}, F1 {:5.2f}\n".format(
                timer.dt(), episode,
                overall_loss_s if args['print_scaled'] else overall_loss,
                overall_acc_s if args['print_scaled'] else overall_acc,
                overall_f1_s if args['print_scaled'] else overall_f1))


            writer.add_scalar('Loss/MacroEval', overall_loss, episode)
            writer.add_scalar('Accuracy/MacroEval', overall_acc, episode)
            writer.add_scalar('F1/MacroEval', overall_f1, episode)

            writer.add_scalar('LossScaled/MacroEval', overall_loss_s, episode)
            writer.add_scalar('AccuracyScaled/MacroEval', overall_acc_s, episode)
            writer.add_scalar('F1Scaled/MacroEval', overall_f1_s, episode)

            writer.flush()

            #####################
            # Best Model Saving #
            #####################
            if overall_acc_s >= best_overall_acc_s:
                for file in os.listdir(checkpoint_save_path):
                    if 'best_model' in file:
                        ep = re.match(r".+macroaccs_\[(.+)\]", file)
                        if float(ep.group(1)):
                            os.remove(os.path.join(checkpoint_save_path, file))

                save_name = "best_model-episode_[{:}]-macroaccs_[{:.2f}].checkpoint".format(episode, overall_acc_s)

                with open(os.path.join(checkpoint_save_path, save_name), 'wb') as f:

                    torch.save(model_init.state_dict(), f)

                print(f"New best scaled accuracy. Saving model as {save_name}\n")
                best_overall_acc_s = overall_acc_s
                curr_patience = args['patience']
            else:
                if episode > args['min_episodes']:
                    curr_patience -= 1
                print(f"Model did not improve with macroaccs_={overall_acc_s}. Patience is now {curr_patience}\n")

            #######################
            # Latest Model Saving #
            #######################
            for file in os.listdir(checkpoint_save_path):
                if 'latest_model' in file:
                    ep = re.match(r".+episode_\[([a-zA-Z0-9\.]+)\].+", file)
                    if ep != None and int(ep.group(1)) <= episode:
                        os.remove(os.path.join(checkpoint_save_path, file))

            save_name = "latest_model-episode_[{:}]-macroaccs_[{:.2f}].checkpoint".format(episode, overall_acc_s)

            with open(os.path.join(checkpoint_save_path, save_name), 'wb') as f:

                torch.save(model_init.state_dict(), f)

            with open(os.path.join(checkpoint_save_path, "latest_trainer.pickle"), 'wb') as f:

                pickle.dump({'episode': episode,
                             'overall_acc_s': overall_acc_s,
                             'best_overall_acc_s': best_overall_acc_s,
                             'curr_patience': curr_patience},
                            f)

            if episode >= args['min_episodes']:
                reduceOnPlateau.step(overall_acc_s)

                curr_lr = meta_optimizer.param_groups[0]['lr']
                if curr_lr < args['min_meta_lr']:
                    print("Patience spent.\nEarly stopping.")
                    raise KeyboardInterrupt

        writer.add_scalar('Meta-lr', meta_optimizer.param_groups[0]['lr'], episode)


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--include', default=['grounded_emotions'], type=str, nargs='+',
                        help='Which datasets to include. Default is all.',
                        choices=['go_emotions', 'crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                        'emotion-cause', 'grounded_emotions', 'ssec', 'tales-emotion', 'tec'])

    # Model hyperparameters
    parser.add_argument('--encoder_name', default='bert-base-uncased', type=str,
                        help='What BERT model to use. Default is bert-base-uncased')

    parser.add_argument('--hidden_dims', default=[128], type=str, nargs='*',
                        help='The hidden dimensions to use. Default is [128]')

    parser.add_argument('--act_fn', default='ReLU', type=str,
                        help='Activation function to use. Default is ReLU',
                        choices=['ReLU', 'Tanh'])

    parser.add_argument('--nu', default=5, type=int,
                        help='Number of levels of the model to freeze. Default is 12')

    parser.add_argument('--dropout', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether or not to apply dropout in BERT during training. Default is False')

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
    parser.add_argument('--n_outer', default=16, type=int,
                        help='Number of outer epochs. Default is 16')

    parser.add_argument('--n_inner', default=5, type=int,
                        help='Number of outer epochs. Default is 7')

    parser.add_argument('--min_episodes', default=2500, type=int,
                        help='Minimum number of episodes. Default is 2500')

    parser.add_argument('--max_episodes', default=3, type=int,
                        help='Maximum number of episodes. Default is 10000')

    parser.add_argument('--eval_every_n', type=int, default=1,
                        help='Number of episodes per evaluation loop.')

    parser.add_argument('--n_eval_per_task', type=int, default=1,
                        help='Number of support sets to evaluate on for a single task.')


    # Optimizer hyperparameters
    parser.add_argument('--meta_lr', default=1e-4, type=float,
                        help='Meta learning rate to use. Default is 1e-4')

    parser.add_argument('--inner_lr', default=1e-3, type=float,
                        help='Inner learning rate to use. Default is 1e-3')

    parser.add_argument('--output_lr', default=1e-1, type=float,
                        help='Outer learning rate to use. Default is 1e-3')

    parser.add_argument('--warmup_steps', default=100, type=float,
                        help='Number of warmup steps. Default is 100')

    parser.add_argument('--clip_val', default=5.0, type=float,
                        help='Value to clip the gradient with. Default is 5.0')

    parser.add_argument('--patience', default=1, type=int,
                        help='Maximum number of evals without improvement before reducing lr learning rate. Default is 1')

    parser.add_argument('--lr_reduce_factor', default=0.1, type=float,
                        help='Meta-learning rate reduction on platuea.')

    parser.add_argument('--min_meta_lr', default=1e-6, type=float,
                        help='Learning rate under which early stopping is induced.')

    # Saving hyperparameters
    parser.add_argument('--checkpoint_path', default='./checkpoints/ProtoMAMLv2', type=str,
                        help='Path where to store the checkpoint. Default is ./checkpoints/ProtoMAML_Rebuild')

    parser.add_argument('--version', default='first_attempt', type=str,
                        help='Name of current run version. Default is debug')

    # Other hyperparameters
    parser.add_argument('--seed', default=610, type=int,
                        help='Random seed.')

    parser.add_argument('--gpu', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether to use a GPU.')

    parser.add_argument('--print_inner_loss', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether or not to print the inner losses. Default is False.')

    parser.add_argument('--print_scaled', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether or not to print metrics scaled to random. Default is True.')

    # parse the arguments
    args = parser.parse_args()
    args = vars(parser.parse_args())

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    for k, v in args.items():
        print(f"{k}: {v}")
    print('-----------------------------\n')


    # train the model
    train(args)
