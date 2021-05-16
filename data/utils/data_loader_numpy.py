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
    def __init__(self, data_subset, k, tokenizer=None, device=None, shuffle=True, max_batch_size=None, verbose=False):
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

        self.verbose = verbose
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
class StratifiedLoaderwClassesSubset():
    def __init__(self, data_subset, k, max_classes=8, tokenizer=None, device=None, shuffle=True, max_batch_size=None, verbose=False):
        """
        Class that acts as dataloader.
        Applies stratified sampling, such that every batch has N (classes) k-shots.
        Samples are strictly non-overlapping.

        Args:
            source_dict (dict): dictionary with source specific data
            split (str): either train or test
            class_to_int (dict): mapping that takes a class str and outputs int
            k (int): number of samples per class
            tokenizer (callable): function that converts list of strings to PyTorch LongTensor.
            shuffle (boolean): whether or not to shuffle the dataset prior to sampling
        """

        self.data_subset = data_subset

        self.n_classes = np.random.randint(2, min(len(data_subset.keys()),
                                                  max_classes)+1)

        self.labels = np.random.choice(list(data_subset.keys()),
                                       self.n_classes, replace=False)
        self.labels = sorted(self.labels)

        self.internal_class_map = {c: i for i,c in enumerate(self.labels)}

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

        self.verbose = verbose
        self.device = 'cpu' if device == None else device
        self.tokenizer = tokenizer

    def _batch(self):

        support_text = np.concatenate(
            [self.data_subset[c][((self.i-1)*self.k):(self.i*self.k)] for c in self.labels])
        support_labels = np.concatenate(
            [[self.internal_class_map[c] for _ in range(self.k)] for c in self.labels])

        query_text = np.concatenate(
            [self.data_subset[c][(self.i*self.k):((self.i+1)*self.k)] for c in self.labels])
        query_labels = np.concatenate(
            [[self.internal_class_map[c] for _ in range(self.k)] for c in self.labels])

        self.i += 2

        return support_labels, support_text, query_labels, query_text

    def __next__(self):

        support_labels, support_text, query_labels, query_text = self._batch()

        if len(support_text) != (self.k * len(self.labels)) or \
            len(query_text) != (self.k * len(self.labels)) or \
                len(support_text) != len(query_text):
            if self.verbose:
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


class RandomTextLoader():
    def __init__(self, tokenizer, batch_size, device=None):

        self.batch_size = batch_size
        self.device = torch.device('cpu') if device == None else device
        self.n_classes = np.random.randint(2, 7)

        self.tokenizer = tokenizer
        self.inv_map = {v: k for k, v in tokenizer.vocab.items()}

    def return_some_random(self):

        sents = []
        for i in range(self.batch_size):
            sent_len = np.random.randint(5, 16)
            sent = ' '.join(
                map(lambda x: self.inv_map[x], np.random.randint(5000, 7000, sent_len)))
            sents.append(sent)

        text = self.tokenizer(list(sents),
                              return_tensors='pt',
                              padding=True).to(self.device)

        labels = torch.LongTensor(np.concatenate([[i for c in range(int(self.batch_size / self.n_classes)+1)]
                                                  for i in range(self.n_classes)]))[:self.batch_size].to(self.device)

        return labels, text

    def __next__(self):

        support_labels, support_text = self.return_some_random()
        query_labels, query_text = self.return_some_random()

        return support_labels, support_text, query_labels, query_text
