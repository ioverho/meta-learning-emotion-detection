from collections import defaultdict
import random
from copy import deepcopy

import numpy as np
import torch

def _data_to_model_input(support_set, query_set, tokenizer, device):
    support_set['text'] = tokenizer(support_set['text'],
                                return_tensors='pt',
                                padding=True).to(device)
    support_set['labels'] = torch.LongTensor(support_set['labels']\
        ).to(device)

    query_set['text'] = tokenizer(query_set['text'],
                                return_tensors='pt',
                                padding=True).to(device)
    query_set['labels'] = torch.LongTensor(query_set['labels']\
        ).to(device)

    return support_set, query_set

class StratifiedLoader():
    def __init__(self, dataset, device, k=16, tokenizer=None, shuffle=True):
        """
        Class that acts as dataloader.
        Applies stratified sampling, such that every batch has N (classes) k-shots.
        Samples are strictly non-overlapping.
        Will raise StopIteration if one of the classes runs out of data to supply.

        Args:
            source_dict (dict): dictionary with source specific data
            split (str): either train or test
            class_to_int (dict): mapping that takes a class str and outputs int
            k (int): number of samples per class
            tokenizer (callable): function that converts list of strings to PyTorch LongTensor.
            shuffle (boolean): whether or not to shuffle the dataset prior to sampling
        """

        self.k = k

        self.data = deepcopy(dataset)

        self.labels = list(self.data.keys())

        self.tokenizer = tokenizer

        if shuffle:
            for c in self.labels:
                random.shuffle(self.data[c])

        self.device = device

    def __next__(self):

        support_set, query_set = defaultdict(list), defaultdict(list)
        for c in self.labels:
            samples = self.data[c][:2*self.k]
            self.data[c] = self.data[c][2*self.k:].extend(samples)
            support_set['text'].extend([s['text'] for s in samples[:self.k]])
            support_set['labels'].extend([s['labels'] for s in samples[:self.k]])

            query_set['text'].extend([s['text'] for s in samples[self.k:]])
            query_set['labels'].extend([s['labels'] for s in samples[self.k:]])

        if self.tokenizer != None:
            support_set, query_set = _data_to_model_input(support_set,\
                query_set, self.tokenizer, self.device)

        return support_set['labels'], support_set['text'],\
            query_set['labels'], query_set['text']

class AdaptiveNKShotLoader():
    def __init__(self, dataset, device, tokenizer=None, max_support_size=128):
        """
        Dataloader with adaptive/stochastic N-way, k-shot batches.
        Support set has random number of examples per class, although proportional to class size.
        Query set is always balanced.
        Not all classes are present if more than 5 classes are present in the dataset.

        Algorithm taken from:
            Triantafillou et al. (2019). Meta-dataset: A dataset of datasets for learning to learn from few examples. arXiv preprint arXiv:1903.03096.
        Steps are as follows:
            1. Sample subset of classes (min 5, max all classes)
            2. Define query set size (max 10 per class)
            3. Define support set size (max 128 for all)
            4. Fill support set with samples, stochastically proportional to support set size
            5. Fill query set with remaining samples

        Args:
            dataset (dict of lists): class separated dictionary with lists of examples
            device (torch.device): which device to push data to
            tokenizer (Huggingface tokenizer, optional): takes list of text and converts to model input. Defaults to None.
            max_support_size (int, optional): max size of the support set. Defaults to 128.
        """

        self.data = dataset
        self.device = device
        self.classes = list(self.data.keys())
        self.tokenizer = tokenizer

        self.max_support_size = max_support_size

    def __next__(self):

        # Compute the N (number of classes)
        if len(self.classes) <= 3:
            n_classes = len(self.classes)
        else:
            n_classes = np.random.randint(low=3, high=len(self.classes))
        classes_sample = np.random.choice(self.classes,
                                          n_classes, replace=False)
        class_lens = np.array([len(self.data[c]) for c in classes_sample])

        # Compute query set size
        # Maximum is 10 per class, per definition, balanced
        min_class_size = np.min(np.floor(0.5 * class_lens))
        query_size = min(10, min_class_size)

        # Compute support set size
        beta = np.random.random()
        class_size_remaining = sum(np.ceil(beta * min(100, len(self.data[c])
                                                      - query_size)) for c in classes_sample)
        support_size = min(self.max_support_size, class_size_remaining)

        # Compute the class specifc k
        alpha = np.random.uniform(low=np.log(
            0.5), high=np.log(2), size=(n_classes))
        R = np.exp(alpha) * class_lens
        R = R / sum(R)

        k_c = np.min(np.stack([np.floor(R * (support_size - n_classes) + 1),
                               class_lens - query_size]), axis=0)

        # Randomly sample both support and query sets
        support_set, query_set = defaultdict(list), defaultdict(list)
        for i, c in enumerate(classes_sample):
            sample = np.random.choice(self.data[c],
                                      size=int(k_c[i] + query_size),
                                      replace=False)

            labels = [s['labels'] for s in sample]
            text = [s['text'] for s in sample]

            support_set['labels'].extend(labels[:int(k_c[i])])
            support_set['text'].extend(text[:int(k_c[i])])

            query_set['labels'].extend(labels[int(k_c[i]):])
            query_set['text'].extend(text[int(k_c[i]):])

        if self.tokenizer != None:
            support_set, query_set = _data_to_model_input(support_set,\
                query_set, self.tokenizer, self.device)

        return support_set['labels'], support_set['text'],\
            query_set['labels'], query_set['text']
