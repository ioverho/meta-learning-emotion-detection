from collections import defaultdict

import torch
import jsonlines

from data.unified_emotion.unified_stratified_loader import UnifiedMetaStratifiedLoader


class unified_emotion():

    def __init__(self, file_path, exclude=['fb-valence-arousal-anon', 'emobank', 'affectivetext', 'emotion-cause', 'electoraltweets'], split_ratio=0.8):
        self.file_path = file_path
        self.exclude = exclude
        self.split_ratio = split_ratio

    def prep(self):

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

                text = line['text']

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

    def lens(self):
        return self.source_lengths

    def get_dataloaders(self, k=4, tokenizer=None):
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
