# imports
import argparse
import os
import json
import random
from timeit import default_timer as timer
import datetime
import torch
from tqdm import tqdm
from transformers import BertTokenizer

# own imports
from data.load_goemotions_data import LoadGoEmotions
from data.load_unifiedemotions_data import LoadUnifiedEmotions
from data.utils.tokenizer import specials
from utils import create_dataloader, handle_epoch_metrics, create_path, initialize_model

# set Huggingface logging to error only
import transformers
transformers.logging.set_verbosity_error()


def perform_step(model, optimizer, batch, device, train=True):
    """
    Function that performs an epoch for the given model.
    Inputs:
        model - BERT model instance
        optimizer - AdamW optimizer instance
        batch - Batch from the dataset to use in the step
        device - PyTorch device to use
        train - Whether to train or test the model
    Outputs:
        loss - Loss of the step
        logits - Predictions of the model
        batch_labels - Real labels of the batch
    """

    # get the features of the batch
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['labels'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)

    # pass the batch through the model
    predictions, loss = model(input_ids, attention_mask=attention_mask, labels=batch_labels, token_type_ids=token_type_ids)

    if train:
        # backward using the loss
        loss.backward()

        # set a step with the optimizer
        optimizer.step()
        optimizer.zero_grad()

    # return the loss, label and prediction
    return loss, predictions, batch_labels


def perform_epoch(args, model, optimizer, dataset, device, train=True):
    """
    Function that performs an epoch for the given model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        optimizer - Optimizer instance
        dataset - Dataset to use
        device - PyTorch device to use
        train - Whether to train or test the model
    Outputs:
        epoch_results - Dictionary containing the average epoch results
    """

    # set model to training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # start a timer for the epoch time
    start_time = timer()

    # initialize dictionary and lists for the results
    result_dict = {}
    epoch_labels = []
    epoch_predictions = []

    # loop over the batches
    if (args.progress_bar):
        dataset = tqdm(dataset)
    for batch in dataset:
        # perform a step for the main task
        step_loss, step_predictions, step_labels = perform_step(model, optimizer, batch, device, train)

        # add the results to the dictionary
        if 'predictions' in result_dict:
            result_dict['predictions'].append(step_predictions)
            result_dict['labels'].append(step_labels)
            result_dict['losses'].append(step_loss)
        else:
            result_dict['predictions'] = [step_predictions]
            result_dict['labels'] = [step_labels]
            result_dict['losses'] = [step_loss]
        epoch_labels.append(step_labels)
        epoch_predictions.append(step_predictions)

    # calculate the loss and accuracy for the different tasks
    epoch_results = handle_epoch_metrics(result_dict, epoch_labels, epoch_predictions)

    # record the end time
    end_time = timer()

    # calculate the elapsed time
    elapsed_time = str(datetime.timedelta(seconds=(end_time - start_time)))

    # add the time to the epoch results
    epoch_results['time'] = {'elapsed_time': elapsed_time}

    # return the epoch results
    return epoch_results


def train_model(args, model, optimizer, train_set, dev_set, test_set, device, path):
    """
    Function for training the model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        optimizer - Optimizer to use
        train_set - Training set
        dev_set - Development set
        test_set - Test set
        device - PyTorch device to use
        path - Path for storing the results
    Outputs:
        model - Trained BERT model instance
        optimizer - Optimizer instance
        gathered_results - Measures of the training process
    """

    print('Starting training..')
    gathered_results = {}

    # evaluate the model before training
    print('Epoch 0:')
    with torch.no_grad():
        dev_results = perform_epoch(args, model, optimizer, dev_set, device, train=False)
    print('Dev results:')
    print(dev_results)

    # save the pre-training evaluation measures
    gathered_results['epoch0'] = {'dev': dev_results}

    # train the model
    best_dev_acc = 0
    epochs_no_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        print('Epoch {}:'.format(epoch))

        # perform a training epoch
        train_results = perform_epoch(args, model, optimizer, train_set, device, train=True)

        # perform a development epoch
        with torch.no_grad():
            dev_results = perform_epoch(args, model, optimizer, dev_set, device, train=False)

        # print the epoch measures
        print('Train results:')
        print(train_results)
        print('Dev results:')
        print(dev_results)

        # save the epoch measures
        gathered_results['epoch' + str(epoch)] = {'train' : train_results, 'dev': dev_results}

        # check whether to save the model or not
        if (round(dev_results['accuracy'], 3) > best_dev_acc):
            epochs_no_improvement = 0
            best_dev_acc = round(dev_results['accuracy'], 3)
            print('Saving new best model..')
            torch.save({
                'epoch': epoch,
                'bert_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(path, "best_model.pt"))
            print('New best model saved')
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement == args.patience:
                print('---')
                break

        print('---')
    print('Training finished')

    # load the best checkpoint
    print('Loading best model..')
    checkpoint = torch.load(os.path.join(path, "best_model.pt"))
    model.load_state_dict(checkpoint['bert_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Best model loaded')

    # return the model, optimizer and results
    return model, optimizer, gathered_results


def main(args):
    """
    Function for handling the arguments and starting the experiment.
    Inputs:
        args - Namespace object from the argument parser
    """

    # set the seed
    torch.manual_seed(args.seed)

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    print('Dataset: {}'.format(args.dataset))
    print('PyTorch device: {}'.format(device))
    print('Max epochs: {}'.format(args.max_epochs))
    print('Patience: {}'.format(args.patience))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Results directory: {}'.format(args.results_dir))
    print('Progress bar: {}'.format(args.progress_bar))
    print('-----------------------------')

    # generate the path to use for the results
    path = create_path(args)
    if not os.path.exists(path):
        os.makedirs(path)

    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=specials())

    # load the datasets
    print('Loading datasets..')
    if args.dataset == 'GoEmotions':
        train_set, dev_set, test_set, num_classes = LoadGoEmotions(args, tokenizer)
    else:
        train_set, dev_set, test_set, num_classes = LoadUnifiedEmotions(args, tokenizer, args.dataset)
    print('Datasets loaded')

    # load the model
    print('Loading model..')
    model, optimizer = initialize_model(args, device, tokenizer, num_classes)
    print('Model loaded')

    # check if a checkpoint is provided
    if args.checkpoint_path is not None:
        # load the model from the given checkpoint
        print('Loading model from checkpoint..')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['bert_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded')
    else:
        # train the model
        model, optimizer, gathered_results = train_model(
            args = args,
            model = model,
            optimizer=optimizer,
            train_set = train_set,
            dev_set = dev_set,
            test_set = test_set,
            device = device,
            path = path
        )

    # test the model
    print('Starting testing..')
    with torch.no_grad():
        test_results = perform_epoch(args, model, optimizer, test_set, device, train=False)
    print('Test results:')
    print(test_results)
    print('Testing finished')

    # save the testing measures
    if args.checkpoint_path is None:
        gathered_results['testing'] = test_results

        # save the results as a json file
        print('Saving results..')
        with open(os.path.join(path, 'results.txt'), 'w') as outfile:
            json.dump(gathered_results, outfile)
        print('Results saved')


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--dataset', default='GoEmotions', type=str,
                        help='What dataset to use. Default is GoEmotions',
                        choices=['GoEmotions', 'crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                            'emotion-cause', 'grounded_emotions', 'ssec', 'tales-emotion', 'tec'])

    # training hyperparameters
    parser.add_argument('--max_epochs', default=20, type=int,
                        help='Maximum number of epochs to train for. Default is 20')
    parser.add_argument('--patience', default=3, type=int,
                        help='Stops training after patience number of epochs without improvement in dev accuracy. Default is 3')

    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate to use for the optimizer. Default is 3e-5')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size. Default is 16')

    # loading hyperparameters
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Path to where the model checkpoint is located. Default is None (no checkpoint used)')

    # other hyperparameters
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--results_dir', default='./baseline_results', type=str,
                        help='Directory where the training results should be created. Default is ./baseline_results')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    # parse the arguments
    args = parser.parse_args()

    # train the model
    main(args)
