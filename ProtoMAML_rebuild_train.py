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

config = {'encoder_name': 'bert-base-cased',
'nu': 12,
'hidden_dims': [128],
'act_fn': 'ReLU',
'include': ['dailydialog'],
'max_support_size': 24,
'n_outer': 16,
'n_inner': 7,
'warmup_steps': 100,
'max_episodes': 10000,
'meta_lr': 1e-4,
'inner_lr': 1e-3,
'output_lr': 1e-3,
'checkpoint_path': './checkpoints/ProtoMAML_Rebuild',
'version': 'single_dataset_wo_dropout_many_episodes',
'gpu': False,
'clip_val': 5.0,
}

def train():
    #######################
    # Logging Directories #
    #######################
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

    ###################
    # Load in dataset #
    ###################
    dataset = unified_emotion("./data/datasets/unified-dataset.jsonl",
                                include=config['include'])
    dataset.prep(text_tokenizer=manual_tokenizer)

    ####################
    # Init models etc. #
    ####################
    model_init = SeqTransformer(config)
    tokenizer = AutoTokenizer.from_pretrained(config['encoder_name'])

    tokenizer.add_special_tokens({'additional_special_tokens': specials()})
    model_init.encoder.model.resize_token_embeddings(len(tokenizer.vocab))

    meta_optimizer = optim.SGD(model_init.parameters(), lr=config['meta_lr'])
    meta_scheduler = get_constant_schedule_with_warmup(meta_optimizer, config['warmup_steps'])

    model_init = model_init.to(device)

    loss_fn = nn.CrossEntropyLoss()

    #######################
    # Overfit query batch #
    #######################
    task = dataset_sampler(dataset, sampling_method='sqrt')

    datasubset = dataset.datasets[task]['train']

    dataloader = StratifiedLoader(datasubset, k=16, shuffle=True, max_batch_size=config['max_support_size'], tokenizer=tokenizer, device=device, classes_subset=False)

    support_labels, support_input, query_labels, query_input = next(dataloader)

    #################
    # Training loop #
    #################
    for episode in range(1, config['max_episodes']+1):

        for ii in range(1, config['n_outer']+1):
            #################
            # Sample a task #
            #################
            task = dataset_sampler(dataset, sampling_method='sqrt')

            datasubset = dataset.datasets[task]['train']

            dataloader = StratifiedLoader(datasubset, k=16, shuffle=True, max_batch_size=config['max_support_size'], tokenizer=tokenizer, device=device, classes_subset=False)


            #####################
            # Create model_task #
            #####################
            for module in model_init.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

            model_task = deepcopy(model_init)
            model_task_optimizer = optim.SGD(model_task.parameters(), lr=config['inner_lr'])
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
            for _ in range(config['n_inner']):

                y = model_task(support_input)
                logits = F.linear(y, W_task, b_task)

                inner_loss = loss_fn(logits, support_labels)

                W_task_grad, b_task_grad = torch.autograd.grad(inner_loss, [W_task, b_task], retain_graph=True)

                inner_loss.backward()

                if config['clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(model_task.parameters(), config['clip_val'])

                model_task_optimizer.step()

                W_task = W_task - config['output_lr'] * W_task_grad
                b_task = b_task - config['output_lr'] * b_task_grad

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
            param.grad = param.grad / config['n_outer']

        if config['clip_val'] > 0:
            torch.nn.utils.clip_grad_norm_(model_init_params, config['clip_val'])

        meta_optimizer.step()
        meta_scheduler.step()
        meta_optimizer.zero_grad()


if __name__ == '__main__':
    for k, v in config.items():
        print(k, v)
    print()
    train()
