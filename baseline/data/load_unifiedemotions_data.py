# imports
from datasets import load_dataset
import jsonlines

# own imports
from utils import create_dataloader
from data.utils.tokenizer import manual_tokenizer

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def PrepareSets(args, tokenizer, label_dict, train_set, dev_set, test_set, first_label=False):
    """
    Function that prepares the datasets for usage.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        label_dict - Dictionary to convert labels to ints
        train_set - Unprepared training set
        dev_set - Unprepared development set
        test_set - Unprepared test set
        first_label - Indicates whether to only use the first label. Default is False
    Outputs:
        train_set - Prepared training set
        dev_set - Prepared development set
        test_set - Prepared test set
    """

    # remove unnecessary columns
    train_set = train_set.remove_columns(['id', 'VAD', 'source', 'text', 'split', 'emotion_model', 'domain', 'labeled', 'annotation_procedure', 'optional'])
    dev_set = dev_set.remove_columns(['id', 'VAD', 'source', 'text', 'split', 'emotion_model', 'domain', 'labeled', 'annotation_procedure', 'optional'])
    test_set = test_set.remove_columns(['id', 'VAD', 'source', 'text', 'split', 'emotion_model', 'domain', 'labeled', 'annotation_procedure', 'optional'])

    # function that creates new instances for all labels
    def handle_multiple_labels(batch):
        new_batch = {'attention_mask': [],
                    'input_ids': [],
                    'labels': [],
                    'token_type_ids': [],
                    'emotions': [],
        }
        for instance_idx, instance in enumerate(batch['emotions']):
            for label in instance:
                if instance[label] == 1:
                    new_batch['attention_mask'].append(batch['attention_mask'][instance_idx])
                    new_batch['input_ids'].append(batch['input_ids'][instance_idx])
                    new_batch['labels'].append(label_dict[label])
                    new_batch['token_type_ids'].append(batch['token_type_ids'][instance_idx])
                    new_batch['emotions'].append(batch['emotions'])
        return new_batch

    # function that takes the first label
    def handle_first_label(batch):
        new_batch = {'attention_mask': [],
                    'input_ids': [],
                    'labels': [],
                    'token_type_ids': [],
                    'emotions': [],
        }
        for instance_idx, instance in enumerate(batch['emotions']):
            for label in instance:
                if instance[label] == 1:
                    new_batch['attention_mask'].append(batch['attention_mask'][instance_idx])
                    new_batch['input_ids'].append(batch['input_ids'][instance_idx])
                    new_batch['labels'].append(label_dict[label])
                    new_batch['token_type_ids'].append(batch['token_type_ids'][instance_idx])
                    new_batch['emotions'].append(batch['emotions'])
                    break
        return new_batch

    # check which label function to use
    if first_label:
        label_fn = handle_first_label
    else:
        label_fn = handle_multiple_labels

    # filter the labels
    train_set = train_set.map(label_fn, batched=True)
    dev_set = dev_set.map(label_fn, batched=True)
    test_set = test_set.map(label_fn, batched=True)

    # remove the emotions columns
    train_set = train_set.remove_columns(['emotions'])
    dev_set = dev_set.remove_columns(['emotions'])
    test_set = test_set.remove_columns(['emotions'])

    # return the prepared datasets
    return train_set, dev_set, test_set


def LoadUnifiedEmotions(args, tokenizer, target_dataset, path="./data/datasets/unified-dataset.jsonl", k_shot=False):
    """
    Function to load the UnifiedEmotions dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        target_dataset - String representing the dataset to load
        path - Path to the unified dataset jsonl file
        k_shot - Indicates whether to make the training set k-shot. Default is False
    Outputs:
        train_set - Training dataset
        dev_set - Development dataset
        test_set - Test dataset
    """

    # load the dataset
    dataset = load_dataset('json', data_files=path)['train']

    # filter out the correct source
    dataset = dataset.filter(lambda example: example['source'] == target_dataset)

    # function that encodes the text
    def encode_text(batch):
        tokenized_batch = tokenizer(batch['text'], padding=True, truncation=True)
        return tokenized_batch

    # tokenize the dataset
    dataset = dataset.map(manual_tokenizer, batched=False)
    dataset = dataset.map(encode_text, batched=False)

    # create a dictionary for converting labels to integers
    label_dict = {}
    for label in dataset[0]['emotions']:
        if dataset[0]['emotions'][label] is not None:
            label_dict[label] = len(label_dict)

    # split the dataset
    if None in dataset['split']:
        # split the dataset into 70% train, 15% test and 15% validation
        dataset = dataset.train_test_split(test_size=0.3, train_size=0.7, shuffle=True)
        train_set = dataset['train']
        dataset = dataset['test'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
        dev_set = dataset['train']
        test_set = dataset['test']
    else:
        train_set = dataset.filter(lambda example: example['split'] == 'train')
        dev_set = dataset.filter(lambda example: example['split'] == 'validation')
        test_set = dataset.filter(lambda example: example['split'] == 'test')

    # create a validation spit for ssec
    if target_dataset == 'ssec':
        test_set = test_set.train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
        dev_set = test_set['train']
        test_set = test_set['test']

    # prepare the data
    train_set, dev_set, test_set = PrepareSets(args, tokenizer, label_dict, train_set, dev_set, test_set)

    # check if k-shot
    if k_shot:
        return train_set, test_set, len(label_dict)

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets and number of classes
    return train_set, dev_set, test_set, len(label_dict)
