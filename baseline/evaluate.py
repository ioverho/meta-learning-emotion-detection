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
from utils import create_dataloader, handle_epoch_metrics, create_path, initialize_model, average_evaluation_results

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


def evaluate_dataset(args, device, tokenizer, train_set, test_set, new_num_classes):
    """
    Function that evaluates on a given dataset.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device instance
        tokenizer - BERT tokenizer instance
        train_set - Dataloader instance containing the training set
        test_set - Dataloader instance containing test set
        new_num_classes - Number classes of the evaluation dataset
    """

    gathered_results = {}

    # get the old number of classes
    task_label_dict = {
        'GoEmotions': 27,
        'crowdflower': 8,
        'dailydialog': 7,
        'electoraltweets': 10,
        'emoint': 4,
        'emotion-cause': 6,
        'grounded_emotions': 2,
        'ssec': 7,
        'tales-emotion': 7,
        'tec': 6,
    }
    old_num_classes = task_label_dict[args.dataset]

    # load the model from the given checkpoint
    print('Loading model from checkpoint..')
    model, optimizer = initialize_model(args, device, tokenizer, old_num_classes)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['bert_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Model loaded')

    # replace the final linear layer
    model.replace_clf(new_num_classes)
    model.to(device)

    # perform a training epoch
    print('Starting training..')
    gathered_results['training'] = {}
    for i in range(1, args.num_epochs + 1):
        print('Epoch ' + str(i) + ':')
        train_results = perform_epoch(args, model, optimizer, train_set, device, train=True)
        gathered_results['training']['epoch' + str(i)] = train_results
        print(train_results)
    print('Training finished')

    # test the model
    print('Starting testing..')
    with torch.no_grad():
        test_results = perform_epoch(args, model, optimizer, test_set, device, train=False)
    print('Test results:')
    print(test_results)
    print('Testing finished')

    # save the testing measures
    gathered_results['testing'] = test_results

    # return the results
    return gathered_results


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
    print('K: {}'.format(args.k))
    print('Number of epochs: {}'.format(args.num_epochs))
    print('Number of runs: {}'.format(args.num_runs))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Results directory: {}'.format(args.results_dir))
    print('Progress bar: {}'.format(args.progress_bar))
    print('-----------------------------')

    # generate the path to use for the results
    path = create_path(args)
    if not os.path.exists(path):
        os.makedirs(path)

    # all evaluation datasets
    eval_datasets = ['GoEmotions', 'crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
        'emotion-cause', 'grounded_emotions', 'ssec', 'tales-emotion', 'tec']

    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=specials())

    # repeat for the different datasets
    all_results = {}
    for dataset_name in eval_datasets:
        dataset_results = {}
        print('Evaluating on ' + dataset_name)

        # load the dataset
        print('Loading datasets..')
        if dataset_name == 'GoEmotions':
            train_set, test_set, new_num_classes = LoadGoEmotions(args, tokenizer, k_shot=True)
        else:
            train_set, test_set, new_num_classes = LoadUnifiedEmotions(args, tokenizer, dataset_name, k_shot=True)
        test_set = create_dataloader(args, test_set, tokenizer)
        print('Datasets loaded')

        # repeat for the specified number of runs
        for run in range(1, args.num_runs + 1):
            print('Run ' + str(run))

            # convert the train set to a k-shot dataloader
            train_loader = create_dataloader(args, train_set, tokenizer, True, new_num_classes)

            # evaluate the model
            results = evaluate_dataset(args, device, tokenizer, train_loader, test_set, new_num_classes)
            dataset_results['run' + str(run)] = results
            print('----')
        all_results[dataset_name] = dataset_results

    # calculate the mean and std for the different runs
    average_results = average_evaluation_results(all_results)
    all_results['average testing'] = average_results

    # save the results as a json file
    print('Saving results..')
    with open(os.path.join(path, 'evaluation_results_k' + str(args.k) + '.txt'), 'w') as outfile:
        json.dump(all_results, outfile)
    print('Results saved')


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--dataset', required=True, type=str,
                        help='Dataset that the current model is trained on, this will be excluded from evaluation. Is required',
                        choices=['GoEmotions', 'crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                            'emotion-cause', 'grounded_emotions', 'ssec', 'tales-emotion', 'tec'])

    # training hyperparameters
    parser.add_argument('--num_runs', default=10, type=int,
                        help='Number of experiment repetitions. Default is 10')
    parser.add_argument('--num_epochs', default=5, type=int,
                        help='Number of epochs for the model to adapt. Default is 5')
    parser.add_argument('--k', default=4, type=int,
                        help='Number of training instances per class. Default is 4')

    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate to use for the optimizer. Default is 3e-5')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size. Default is 16')

    # loading hyperparameters
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Path to where the model checkpoint is located. Is required')

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
