import numpy as np
import torch

def _data_to_model_input(support_labels, support_text, query_labels, query_text, tokenizer, device):

    support_labels = torch.LongTensor(support_labels).to(device)

    support_text = tokenizer(list(support_text),
                             return_tensors='pt',
                             padding=True).to(device)

    query_labels = torch.LongTensor(query_labels).to(device)

    query_text = tokenizer(list(query_text),
                           return_tensors='pt',
                           padding=True).to(device)

    return support_labels, support_text, query_labels, query_text

class StratifiedLoader():
    def __init__(self, data_subset, k, tokenizer=None, device=None, shuffle=True, max_batch_size=None, classes_subset=False, verbose=False):
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

        self.data_subset = data_subset
        self.labels = list(self.data_subset.keys())

        if classes_subset:
            self.n_classes = min(np.random.randint(2, len(self.labels)),
                                len(self.labels))
            np.random.choice(self.labels, size=self.n_classes)
        else:
            self.n_classes = len(self.labels)

        if shuffle:
            for c in self.labels:
                np.random.shuffle(self.data_subset[c])

        self.k = k
        if max_batch_size != None and (self.k * len(self.labels)) > max_batch_size:
            transgression = max_batch_size / (self.k * len(self.labels))
            self.k = int(transgression * self.k)
            if verbose:
                print(f"Maximum batch size exceeded. Limiting k to {self.k}.")

        self.i = 1

        self.device = 'cpu' if device == None else device
        self.tokenizer = tokenizer

    def _batch(self):

        support_text = np.concatenate(
            [self.data_subset[c][((self.i-1)*self.k):(self.i*self.k)] for c in self.labels])
        support_labels = np.concatenate(
            [[c for _ in range(self.k)] for c in self.labels])

        query_text = np.concatenate(
            [self.data_subset[c][(self.i*self.k):((self.i+1)*self.k)] for c in self.labels])
        query_labels = np.concatenate(
            [[c for _ in range(self.k)] for c in self.labels])

        self.i += 2

        return support_labels, support_text, query_labels, query_text

    def __next__(self):

        support_labels, support_text, query_labels, query_text = self._batch()

        if len(support_text) != (self.k * len(self.labels)) or \
            len(query_text) != (self.k * len(self.labels)) or \
                len(support_text) != len(query_text):
            print("No longer able to generate stratified sample.\
                Reshuffling and resampling.")
            for c in self.labels:
                np.random.shuffle(self.data_subset[c])
            self.i = 1

            support_labels, support_text, query_labels, query_text = self._batch()

        if self.tokenizer != None:
            return _data_to_model_input(support_labels, support_text,
                                        query_labels, query_text,
                                        self.tokenizer, self.device)
        else:
            return support_labels, support_text, query_labels, query_text
