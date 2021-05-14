# imports
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# huggingface imports
import transformers
from transformers import BertTokenizer
from transformers import AdamW
from transformers.data.data_collator import DataCollatorWithPadding

# own imports
from models.custombert import CustomBERT


def create_path(args):
    """
    Function that creates a path for the results based on the model arguments.
    Inputs:
        args - Namespace object from the argument parser
    Outputs:
        path - Path where to store the results
    """

    # create the path
    path = os.path.join(
        args.results_dir,
        args.dataset
    )

    # return the path
    return path


def initialize_model(args, device, tokenizer, num_classes):
    """
    Function that initializes the model, tokenizer and optimizer.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to use
        tokenizer - BERT tokenizer instance
        num_classes - Number of classes of the dataset
    Outputs:
        model - MultiTask BERT model instance
        optimizer - Optimizer instance
    """

    # load the model
    model = CustomBERT(num_classes).to(device)
    model.encoder.model.resize_token_embeddings(len(tokenizer.vocab) + 3)

    # create the optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # return the model, tokenizer and optimizer
    return model, optimizer


def create_dataloader(args, dataset, tokenizer):
    """
    Function to create a PyTorch Dataloader from a given dataset.
    Inputs:
        args - Namespace object from the argument parser
        dataset - Dataset to convert to Dataloader
        tokenizer - BERT tokenizer instance
    Outputs:
        dataset - DataLoader object of the dataset
    """

    # create a data collator function
    data_collator = DataCollatorWithPadding(tokenizer)

    # create the dataloader
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        drop_last=False,
        shuffle=True,
    )

    # return the dataset
    return dataset


def compute_accuracy_f1(preds, labels):
    """
    Function that calculates the accuracy and f1 scores.
    Inputs:
        preds - List of batched predictions from the model
        labels - List of batched real labels
    Outputs:
        acc - Accuracy of the predictions and real labels
        f1 - F1 scores of the predictions and real labels
    """

    # concatenate the predictions and labels
    preds = torch.cat(preds, dim=0).squeeze()
    labels = torch.cat(labels, dim=0).squeeze()

    # check if regression or classification
    if len(preds.shape) > 1:
        preds = torch.nn.functional.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)
    else:
        preds = torch.round(preds)
        labels = torch.round(labels)

    # calculate the accuracy
    acc = accuracy_score(labels.cpu().detach(), preds.cpu().detach())

    # round to 4 decimals
    acc = round(acc, 4)

    # compute the f1 scores
    f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average=None).tolist()

    # return the accuracy and f1 scores
    return acc, f1


def handle_epoch_metrics(step_metrics, epoch_labels, epoch_predictions):
    """
    Function that handles the metrics per epoch.
    Inputs:
        step_metrics - Dictionary containing the results of the steps of an epoch
        epoch_labels - List of labels from the different steps
        epoch_predictions - List of predictions from the different steps
    Outputs:
        epoch_merics - Dictionary containing the averaged results of an epoch
    """

    # compute the loss
    loss = torch.mean(torch.stack(step_metrics['losses'], dim=0), dim=0)
    loss = round(loss.item(), 4)

    # compute the accuracy and f1
    accuracy, f1 = compute_accuracy_f1(step_metrics['predictions'], step_metrics['labels'])

    # create a new epoch dictionary
    epoch_metrics = {'loss': loss, 'accuracy': accuracy, 'f1': f1}

    # return the epoch dictionary
    return epoch_metrics
