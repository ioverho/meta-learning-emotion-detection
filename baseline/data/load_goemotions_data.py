# imports
from datasets import load_dataset

# own imports
from utils import create_dataloader
from data.utils.tokenizer import manual_tokenizer

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def PrepareSets(args, tokenizer, train_set, dev_set, test_set, first_label=False):
    """
    Function that prepares the datasets for usage.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        train_set - Unprepared training set
        dev_set - Unprepared development set
        test_set - Unprepared test set
        first_label - Indicates whether to only use the first label. Default is False
    Outputs:
        train_set - Prepared training set
        dev_set - Prepared development set
        test_set - Prepared test set
    """

    # filter out al instances where the emotion is neutral
    train_set = train_set.filter(lambda example: not 27 in example['labels'])
    dev_set = dev_set.filter(lambda example: not 27 in example['labels'])
    test_set = test_set.filter(lambda example: not 27 in example['labels'])

    # remove unnecessary columns
    train_set = train_set.remove_columns(['text', 'id'])
    dev_set = dev_set.remove_columns(['text', 'id'])
    test_set = test_set.remove_columns(['text', 'id'])

    # function that creates new instances for all labels
    def handle_multiple_labels(batch):
        new_batch = {'attention_mask': [],
                    'input_ids': [],
                    'labels': [],
                    'token_type_ids': [],
        }
        for instance_idx, instance in enumerate(batch['labels']):
            for label in instance:
                new_batch['attention_mask'].append(batch['attention_mask'][instance_idx])
                new_batch['input_ids'].append(batch['input_ids'][instance_idx])
                new_batch['labels'].append(label)
                new_batch['token_type_ids'].append(batch['token_type_ids'][instance_idx])
        return new_batch

    # function that takes the first label
    def handle_first_label(batch):
        batch['labels'] = batch['labels'][0]
        return batch

    # check which label function to use
    if first_label:
        label_fn = handle_first_label
        batched = False
    else:
        label_fn = handle_multiple_labels
        batched = True

    # filter the labels
    train_set = train_set.map(label_fn, batched=batched)
    dev_set = dev_set.map(label_fn, batched=batched)
    test_set = test_set.map(label_fn, batched=batched)

    # return the prepared datasets
    return train_set, dev_set, test_set


def LoadGoEmotions(args, tokenizer, first_label=False):
    """
    Function to load the GoEmotions dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        first_label - Indicates whether to only use the first label. Default is False
    Outputs:
        train_set - Training dataset
        dev_set - Development dataset
        test_set - Test dataset
    """

    # load the dataset
    dataset = load_dataset("go_emotions", "simplified")

    # function that encodes the text
    def encode_text(batch):
        tokenized_batch = tokenizer('[CLS] ' + batch['text'] + ' [SEP]', padding=True, truncation=True)
        return tokenized_batch

    # tokenize the dataset
    dataset = dataset.map(manual_tokenizer, batched=False)
    dataset = dataset.map(encode_text, batched=False)

    # split into test, dev and train
    train_set = dataset['train']
    dev_set = dataset['validation']
    test_set = dataset['test']

    # prepare the data
    train_set, dev_set, test_set = PrepareSets(args, tokenizer, train_set, dev_set, test_set, first_label)

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets and number of classes
    return train_set, dev_set, test_set, 27
