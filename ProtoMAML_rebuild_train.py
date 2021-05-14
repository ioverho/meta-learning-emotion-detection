import os
from copy import deepcopy
import pickle

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from models.seqtransformer import SeqTransformer
from data.unified_emotion_numpy import unified_emotion
from data.utils.tokenizer import manual_tokenizer, specials
from data.utils.data_loader_numpy import StratifiedLoader
from data.utils.sampling import dataset_sampler
from utils.metrics import logging_metrics


def train(args):
    #######################
    # Logging Directories #
    #######################
    log_dir = os.path.join(args.checkpoint_path, args.version)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoint'), exist_ok=True)
    print(f"Saving models and logs to {log_dir}")

    with open(os.path.join(log_dir, 'checkpoint', 'hparams.pickle'), 'wb') as file:
        pickle.dump(args, file)

    ## Initialization
    # Device
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    # Build the tensorboard writer
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))

    ###################
    # Load in dataset #
    ###################
    dataset = unified_emotion("./data/datasets/unified-dataset.jsonl",
                                include=args.include)
    dataset.prep(text_tokenizer=manual_tokenizer)

    ####################
    # Init models etc. #
    ####################
    model_init = SeqTransformer(args)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    tokenizer.add_special_tokens({'additional_special_tokens': specials()})
    model_init.encoder.model.resize_token_embeddings(len(tokenizer.vocab))

    meta_optimizer = optim.SGD(model_init.parameters(), lr=args.meta_lr)
    meta_scheduler = get_constant_schedule_with_warmup(meta_optimizer, args.warmup_steps)

    model_init = model_init.to(device)

    loss_fn = nn.CrossEntropyLoss()

    #######################
    # Overfit query batch #
    #######################
    task = dataset_sampler(dataset, sampling_method='sqrt')

    datasubset = dataset.datasets[task]['train']

    dataloader = StratifiedLoader(datasubset, k=16, shuffle=True, max_batch_size=args.max_support_size, tokenizer=tokenizer, device=device, classes_subset=False)

    support_labels, support_input, query_labels, query_input = next(dataloader)

    #################
    # Training loop #
    #################
    for episode in range(1, args.max_episodes+1):

        for ii in range(1, args.n_outer+1):
            #################
            # Sample a task #
            #################
            task = dataset_sampler(dataset, sampling_method='sqrt')

            datasubset = dataset.datasets[task]['train']

            dataloader = StratifiedLoader(datasubset, k=16, shuffle=True, max_batch_size=args.max_support_size, tokenizer=tokenizer, device=device, classes_subset=False)


            #####################
            # Create model_task #
            #####################
            for module in model_init.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

            model_task = deepcopy(model_init)
            model_task_optimizer = optim.SGD(model_task.parameters(), lr=args.inner_lr)
            #model_task.train()
            model_task.zero_grad()

            #######################
            # Generate prototypes #
            #######################
            support_labels, support_input, query_labels, query_input = next(dataloader)
            #support_labels, support_input, _, _ = next(dataloader)

            #model_init.train()

            y = model_init(support_input)

            labs = torch.sort(torch.unique(support_labels))[0]
            prototypes = torch.stack([torch.mean(y[support_labels==c], dim=0) for c in labs])

            W_init = 2 * prototypes
            b_init = -torch.norm(prototypes, p=2, dim=1)

            W_task, b_task = W_init.detach(), b_init.detach()
            W_task.requires_grad, b_task.requires_grad = True, True

            #################
            # Adapt to data #
            #################
            for _ in range(args.n_inner):

                y = model_task(support_input)
                logits = F.linear(y, W_task, b_task)

                inner_loss = loss_fn(logits, support_labels)

                W_task_grad, b_task_grad = torch.autograd.grad(inner_loss, [W_task, b_task], retain_graph=True)

                inner_loss.backward()

                if args.clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model_task.parameters(), args.clip_val)

                model_task_optimizer.step()

                W_task = W_task - args.output_lr * W_task_grad
                b_task = b_task - args.output_lr * b_task_grad

                print(f"\tInner Loss: {inner_loss.detach().cpu().item()}")

            #########################
            # Validate on query set #
            #########################
            for module in model_task.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

            W_task = W_init + (W_task - W_init).detach()
            b_task = b_init + (b_task - b_init).detach()

            y = model_task(query_input)
            logits = F.linear(y, W_task, b_task)

            outer_loss = loss_fn(logits, query_labels)

            model_task_params = [param for param in model_task.parameters() if param.requires_grad]
            model_task_grads = torch.autograd.grad(outer_loss, model_task_params, retain_graph=True)

            model_init_params = [param for param in model_init.parameters() if param.requires_grad]
            model_init_grads = torch.autograd.grad(outer_loss, model_init_params, retain_graph=False)

            model_init_grads = model_init_grads + model_task_grads

            for param, grad in zip(model_init_params, model_init_grads):
                if param.grad != None:
                    param.grad += grad.detach()
                else:
                    param.grad = grad.detach()

            with torch.no_grad():
                mets = logging_metrics(logits.detach().cpu(), query_labels.detach().cpu())
                outer_loss_ = outer_loss.detach().cpu().item()
                acc = mets['acc']
                f1 = mets['f1']

                #print(torch.norm(W_task - W_init, p=2), sum((x - y).abs().sum() for x, y in zip(model_init.state_dict().values(), model_task.state_dict().values())))

            print("Train | Episode {:}.{:} | Task {:<20s}, N={:} | Loss {:.4E}, Acc {:5.2f}, F1 {:5.2f} | Mem {:5.2f} GB".format(
                episode, ii, task, dataloader.n_classes,
                outer_loss_, acc, f1,
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))

            writer.add_scalars('Loss/Train', {'task': outer_loss_}, episode)
            writer.add_scalars('Accuracy/Train', {'task': acc}, episode)
            writer.add_scalars('F1/Train', {'task': f1}, episode)

            writer.flush()

        print('')

        ############################
        # Init Model Backward Pass #
        ############################
        model_init_params = [param for param in model_init.parameters() if param.requires_grad]
        for param in model_init_params:
            param.grad = param.grad / args.n_outer

        if args.clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model_init_params, args.clip_val)

        meta_optimizer.step()
        meta_scheduler.step()
        meta_optimizer.zero_grad()


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--encoder_name', default='bert-base-uncased', type=str,
                        help='What BERT model to use. Default is bert-base-uncased')
    parser.add_argument('--hidden_dims', default=[128], type=str, nargs='*',
                        help='The hidden dimensions to use. Default is [128]')
    parser.add_argument('--act_fn', default='ReLU', type=str,
                        help='Activation function to use. Default is ReLU',
                        choices=['ReLU', 'Tanh'])
    parser.add_argument('--max_support_size', default=24, type=int,
                        help='Maximum support size. Default is 24')
    parser.add_argument('--include', default=['dailydialog'], type=str, nargs='*',
                        help='Which datasets to include. Default is [dailydialog]',
                        choices=['GoEmotions', 'crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                            'emotion-cause', 'grounded_emotions', 'ssec', 'tales-emotion', 'tec'])

    # training hyperparameters
    parser.add_argument('--nu', default=12, type=int,
                        help='Number of levels of the model to freeze. Default is 12')
    parser.add_argument('--n_outer', default=16, type=int,
                        help='Number of outer epochs. Default is 16')
    parser.add_argument('--n_inner', default=7, type=int,
                        help='Number of outer epochs. Default is 7')
    parser.add_argument('--max_episodes', default=10000, type=int,
                        help='Maximum number of episodes. Default is 10000')

    # optimizer hyperparameters
    parser.add_argument('--meta_lr', default=1e-4 type=float,
                        help='Meta learning rate to use. Default is 1e-4')
    parser.add_argument('--inner_lr', default=1e-3 type=float,
                        help='Inner learning rate to use. Default is 1e-3')
    parser.add_argument('--outer_lr', default=1e-3 type=float,
                        help='Outer learning rate to use. Default is 1e-3')
    parser.add_argument('--warmup_steps', default=100, type=float,
                        help='Number of warmup steps. Default is 100')
    parser.add_argument('--clip_val', default=5.0 type=float,
                        help='Value to clip the gradient with. Default is 5.0')

    # saving hyperparameters
    parser.add_argument('--checkpoint_path', default='./checkpoints/ProtoMAML_Rebuild', type=str,
                        help='Path where to store the checkpoint. Default is ./checkpoints/ProtoMAML_Rebuild')
    parser.add_argument('--version', default='single_dataset_wo_dropout_many_episodes', type=str,
                        help='Name of current run version. Default is single_dataset_wo_dropout_many_episodes')

    # other hyperparameters
    parser.add_argument('--gpu', action='store_true', help='Whether to use a GPU.')

    # parse the arguments
    args = parser.parse_args()

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    print('Encoder name: {}'.format(args.encoder_name))
    print('Hidden dimensions: {}'.format(args.hidden_dims))
    print('Activation function: {}'.format(args.act_fn))
    print('Max support size: {}'.format(args.max_support_size))
    print('Include: {}'.format(args.include))
    print('Nu: {}'.format(args.nu))
    print('Num outer: {}'.format(args.n_outer))
    print('Num inner: {}'.format(args.n_inner))
    print('Max episodes: {}'.format(args.max_episodes))
    print('Meta learning rate: {}'.format(args.meta_lr))
    print('Inner learning rate: {}'.format(args.inner_lr))
    print('Outer learning rate: {}'.format(args.outer_lr))
    print('Warmup steps: {}'.format(args.warmup_steps))
    print('Gradient clipping value: {}'.format(args.clip_val))
    print('checkpoint path: {}'.format(args.checkpoint_path))
    print('Version: {}'.format(args.version))
    print('use GPU: {}'.format(args.gpu))
    print('-----------------------------')

    # train the model
    train(args)
