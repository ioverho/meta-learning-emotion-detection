import random
from copy import deepcopy

import numpy as np
import torch

class MetaStratifiedLoader():
    def __init__(self, source_dict, split, class_to_int, k, tokenizer, shuffle, device):
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
        self.split = split

        self.class_to_int = class_to_int
        self.int_to_class = {v: k for k, v in class_to_int.items()}

        self.data = deepcopy(source_dict[split])

        self.labels = list(self.data.keys())

        self.tokenizer = tokenizer

        if shuffle:
            for c in self.labels:
                random.shuffle(self.data[c])

        self.device = device

    def lens(self):
        return [len(self.data[k]) for k in self.labels]

    def __len__(self):
        return sum(self.lens())

    def __next__(self):

        if not all([l > self.k for l in self.lens()]):
            raise StopIteration("Some classes ran out of data.")

        text = []
        labels = []
        for c in self.labels:
            samples, self.data[c] = self.data[c][:self.k], self.data[c][self.k:]
            text.extend([s['text'] for s in samples])
            labels.extend([s['labels'] for s in samples])

        if self.tokenizer == None:
            return labels, text
        else:
            encoded = self.tokenizer(text, padding=True,
                                     return_tensors="pt").to(self.device)
            text = encoded['input_ids']
            mask = encoded['attention_mask']

            labels = torch.LongTensor(labels).to(self.device)

            return labels, text, mask
