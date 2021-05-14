from collections import defaultdict

import numpy as np
import datasets
from datasets import load_dataset

class go_emotions():

    def __init__(self, first_label_only=False, verbose=True):
        """
        Class for the 'GoEmotions Dataset'.

        Args:
            first_label_only (boolean, optional): boolean indicating whether to only use the first label when multiple labels are present. Defaults to True
        """

        self.dataset_name = 'go_emotions'
        self.first_label_only = first_label_only
        self.verbose = verbose

    def prep(self, text_tokenizer=lambda x: x, text_tokenizer_kwargs=dict()):
        """
        Generates dataset from the Huggingface dataset.

        Args:
            text_tokenizer (callable, optional): function that processes a line of text. Defaults to identity (raw text).
        """

        dataset = load_dataset('go_emotions')

        train_set = dataset['train']
        dev_set = dataset['validation']
        test_set = dataset['test']

        datasets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        source_lengths = dict()
        label_map = defaultdict()
        label_map["admiration"] = 0
        label_map["amusement"] = 1
        label_map["anger"] = 2
        label_map["annoyance"] = 3
        label_map["approval"] = 4
        label_map["caring"] = 5
        label_map["confusion"] = 6
        label_map["curiosity"] = 7
        label_map["desire"] = 8
        label_map["disappointment"] = 9
        label_map["disapproval"] = 10
        label_map["disgust"] = 11
        label_map["embarrassment"] = 12
        label_map["excitement"] = 13
        label_map["fear"] = 14
        label_map["gratitude"] = 15
        label_map["grief"] = 16
        label_map["joy"] = 17
        label_map["love"] = 18
        label_map["nervousness"] = 19
        label_map["optimism"] = 20
        label_map["pride"] = 21
        label_map["realization"] = 22
        label_map["relief"] = 23
        label_map["remorse"] = 24
        label_map["sadness"] = 25
        label_map["surprise"] = 26
        label_map["neutral"] = 27
        inv_label_map = {v: k for k, v in label_map.items()}

        for set in [('train', train_set), ('validation', dev_set), ('test', test_set)]:
            split_name = set[0]
            split_dataset = set[1]
            for instance in split_dataset:
                id = source_lengths.get('go_emotions', 0)

                text = text_tokenizer(
                    instance['text'], **text_tokenizer_kwargs)
                text = text.encode('utf-8').decode('ascii', 'ignore')
                if text == None:
                    continue
                if isinstance(text, list):
                    text = ' '.join(text)

                # check how to handle multiple labels
                labels = instance['labels']
                if (len(labels) > 1) and not self.first_label_only:
                    for label_idx, label in enumerate(labels):
                        label = inv_label_map[label]
                        datasets['go_emotions'][split_name][label].append(text)
                    source_lengths['go_emotions'] = id + len(labels)
                else:
                    label = inv_label_map[labels[0]]
                    source_lengths['go_emotions'] = id + 1

                datasets['go_emotions'][split_name][label].append(text)

        total_removed, total_data_removed = 0, 0
        removing = []
        for source in datasets.keys():
            n_classes = len(datasets[source]['train'].keys())
            for c in datasets[source]['train'].keys():
                train_size = len(datasets[source]['train'][c])
                valid_size = len(datasets[source]['validation'][c])
                test_size = len(datasets[source]['test'][c])

                keep = (train_size >= 96 and test_size >= 64)

                if (not keep):
                    if self.verbose:
                        print("Removed {:}/{:} for too little data |train|={}, |valid|={}, |test|={}".
                              format(source, c, train_size, valid_size, test_size))
                        #self.inv_label_map[source][c]
                    total_removed += 1
                    total_data_removed += train_size + test_size

                    source_lengths[source] -= train_size + test_size

                    removing.append((source, c))

        for source, c in removing:
            del datasets[source]['train'][c]
            del datasets[source]['validation'][c]
            del datasets[source]['test'][c]

        if self.verbose:
            print("Removed a total of {:} classes and {:} examples.".format(
                total_removed, total_data_removed))

        for source in datasets.keys():
            assert len(datasets[source]['train'].keys()) >= 2, print(
                f"{source} has too few classes remaining.")

        for source in datasets.keys():
            for split in datasets[source].keys():
                for c in datasets[source][split].keys():
                    datasets[source][split][c] = np.stack(
                        datasets[source][split][c])

        label_map = dict()
        inv_label_map = dict()
        for source in datasets.keys():
            label_map[source] = {emotion: i for i, emotion in
                                 enumerate(sorted(datasets[source]
                                                  ['train'].keys()))}
            inv_label_map[source] = {v: k for k,
                                     v in label_map[source].items()}

        for source in datasets.keys():
            for split in datasets[source].keys():
                datasets[source][split] = {
                    label_map[source][c]: v for c, v in datasets[source][split].items()}

        self.datasets = datasets
        self.source_lengths = source_lengths
        self.label_map = label_map
        self.inv_label_map = inv_label_map

    @property
    def lens(self):
        """Lengths of the individual datasets
        """
        return self.source_lengths

    def __getitem__(self, i):
        return self.datasets.get(i, None)
