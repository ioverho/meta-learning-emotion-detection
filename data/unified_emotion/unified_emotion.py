from collections import defaultdict

import torch
import jsonlines

from data.unified_emotion.unified_stratified_loader import UnifiedMetaStratifiedLoader

class unified_emotion():
    """Class for the 'Unified Emotion Dataset'. Data from https://github.com/sarnthil/unify-emotion-datasets.
    """

    def __init__(self, file_path, exclude=['fb-valence-arousal-anon', 'emobank', 'affectivetext', 'emotion-cause', 'electoraltweets'], split_ratio=0.8):
        """
        Class for the 'Unified Emotion Dataset'.
        Data from https://github.com/sarnthil/unify-emotion-datasets.
        By default includes the following:
            - crowdflower_data, 40k tweets, 14 labels
            - dailydialog, 13k dialogs, 6 labels
            - emotiondata-aman, 15k sents, 7 labels
            - grounded_emotions, 2.5k tweets, 2 labels
            - isear, 3000 docs, 7 labels
            - tales-emotions, 15k sents, 8 labels
            - emoint

        Args:
            file_path (str): path to the 'unified-dataset.jsonl' file
            exclude (list, optional): tasks to exclude. Defaults to ['fb-valence-arousal-anon', 'emobank', 'affectivetext', 'emotion-cause', 'electoraltweets'].
            split_ratio (float, optional): amount of data reserved for test sets. Defaults to 0.8.
        """
        self.file_path = file_path
        self.exclude = exclude
        self.split_ratio = split_ratio

    def prep(self, text_tokenizer=lambda x: x):
        """Generates dataset from unified file.

        Args:
            text_tokenizer (callable, optional): function that processes a line of text. Defaults to identity (raw text).
        """

        datasets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        source_lengths = dict()
        label_map = defaultdict()

        with jsonlines.open(self.file_path) as file:
            for i, line in enumerate(file.iter()):

                source = line['source']
                if source in self.exclude:
                    continue

                split = 'all' if line.get('split', None) == None else line['split']

                id = source_lengths.get(source, 0)

                labels = {k: v for k, v in sorted(line['emotions'].items()) if v != None}
                if id == 0:
                    label_map[source] = {k: i for i, (k, _) in enumerate(labels.items())}
                label = label_map[source][max(labels, key=labels.get)]

                text = text_tokenizer(line['text'])

                datasets[source][split][label].append({'idx': id, 'labels': label, 'text': text})
                source_lengths[source] = id + 1

        for source in datasets.keys():
            if len(datasets[source].keys()) == 1 and 'all' in datasets[source].keys():
                class_lengths = {k: len(datasets[source]['all'][k]) for k in datasets[source]['all'].keys()}
                for c, l in class_lengths.items():
                    train_l = int(self.split_ratio * l)
                    datasets[source]['train'][c] = datasets[source]['all'][c][:train_l]
                    datasets[source]['test'][c] = datasets[source]['all'][c][train_l:]

                del datasets[source]['all']

        self.datasets = datasets
        self.source_lengths = source_lengths
        self.label_map = label_map

    @property
    def lens(self):
        """Lengths of the individual datasets
        """
        return self.source_lengths

    def get_dataloaders_all(self, k=4, tokenizer=None):
        trainloaders = []
        testloaders = []
        for source in self.datasets.keys():
            for split in self.datasets[source].keys():
                source_dict = self.datasets[source]
                dataloader = UnifiedMetaStratifiedLoader(source_dict=source_dict,
                                                         split=split,
                                                         class_to_int=self.label_map[source],
                                                         k=k,
                                                         tokenizer=tokenizer
                                                         )

                if split == 'train':
                    trainloaders.append((source, dataloader))
                else:
                    testloaders.append((source, dataloader))

        return trainloaders, testloaders

    def get_dataloader(self, source_name,  k=4, tokenizer=None, shuffle=True):
        """Generates a dataloader from a specified dataset.
        See UnifiedMetaStratifiedLoader for more.

        Args:
            source_name (str): a dataset from one of the processed ones.
            k (int, optional): the k-shot. Defaults to 4.
            tokenizer (callable, optional): function that processes list of strings to PyTorch tensor. Defaults to None.
            shuffle (boolean, optional): whether or not to shuffle the train data. Defaults to True.

        Returns:
            dataloaders: iterable of data_loaders. First is train, last is test.
        """
        data_loaders = []
        for split in self.datasets[source_name].keys():
            source_dict = self.datasets[source_name]
            dataloader = UnifiedMetaStratifiedLoader(source_dict=source_dict,
                                                        split=split,
                                                        class_to_int=self.label_map[source_name],
                                                        k=k,
                                                        tokenizer=tokenizer,
                                                        shuffle=shuffle if split=='train' else False
                                                        )

            if split == 'train':
                data_loaders.insert(0, dataloader)
            else:
                data_loaders.append(dataloader)

        return data_loaders
