from collections import defaultdict

import torch
import jsonlines


def unified_emotion_info():
    return [{'source': 'affectivetext', 'size': 250, 'domain': 'headlines', 'classes': 6, 'special': 'non-discrete, multiple labels'},
            {'source': 'crowdflower', 'size': 40000, 'domain': 'tweets', 'classes': 14, 'special': 'includes no-emotions class'},
            {'source': 'dailydialog', 'size': 13000, 'domain': 'conversations', 'classes': 6, 'special': 'includes no-emotions class'},
            {'source': 'electoraltweets', 'size': 4058, 'domain': 'tweets', 'classes': 8, 'special': 'includes no-emotions class'},
            {'source': 'emobank', 'size': 10000, 'domain': 'headlines', 'classes': 3, 'special': 'VAD regression'},
            {'source': 'emoint', 'size': 7097, 'domain': 'tweets', 'classes': 6, 'special': 'annotated by experts'},
            {'source': 'emotion-cause', 'size': 2414, 'domain': 'artificial', 'classes': 6, 'special': 'N/A'},
            {'source': 'fb-valence-arousal-anon', 'size': 2800, 'domain': 'facebook', 'classes': 3, 'special': 'VA regression'},
            {'source': 'grounded_emotions', 'size': 2500, 'domain': 'tweets', 'classes': 2, 'special': 'N/A'},
            {'source': 'ssec', 'size': 4868, 'domain': 'tweets', 'classes': 8, 'special': 'multiple labels per sentence'},
            {'source': 'tales-emotion', 'size': 15302, 'domain': 'fairytales', 'classes': 6, 'special': 'includes no-emotions class'},
            {'source': 'tec', 'size': 21051, 'domain': 'tweets', 'classes': 7, 'special': 'annotated by experts'}
            ]

class unified_emotion():
    """Class for the 'Unified Emotion Dataset'. Data from https://github.com/sarnthil/unify-emotion-datasets.
    """

    def __init__(self, file_path, include=['grounded_emotions'], split_ratio=0.8, verbose=False, first_label_only=False):
        """
        Class for the 'Unified Emotion Dataset'.
        Data from https://github.com/sarnthil/unify-emotion-datasets.

        Args:
            file_path (str): path to the 'unified-dataset.jsonl' file
            include (list, optional): if not None, will only use the datasets in the include list. Defaults to None
            exclude (list, optional): tasks to exclude. Defaults to ['fb-valence-arousal-anon', 'emobank', 'affectivetext', 'emotion-cause', 'electoraltweets'].
            split_ratio (float, optional): amount of data reserved for test sets. Defaults to 0.8.
        """
        self.file_path = file_path
        self.include = include
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.first_label_only = first_label_only

        self.info = [row for row in unified_emotion_info() if row['source'] in self.include]

    def prep(self, text_tokenizer=lambda x: x, text_tokenizer_kwargs=dict()):
        """Generates dataset from unified file.

        Args:
            text_tokenizer (callable, optional): function that processes a line of text. Defaults to identity (raw text).
        """

        datasets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        source_lengths = dict()
        label_map = defaultdict()

        with jsonlines.open(self.file_path) as file:
            for i, line in enumerate(file.iter()):

                # Skip line if not in include list
                source = line['source']
                if not source in self.include:
                    continue

                # Give 'all' split if data doesn't have its own train/test split
                split = 'all' if line.get('split', None) == None else line['split']

                # Give line a data specific id
                id = source_lengths.get(source, 0)

                # Convert the labels
                # Saves the mapping if this is the first line of a dataset
                labels = {k: v for k, v in sorted(line['emotions'].items())
                          if v != None}
                if id == 0:
                    label_map[source] = {k: i for i,
                                        (k, _) in enumerate(labels.items())}

                # All present emotions (labels > 1)
                present_emotions = [emotion for emotion,
                                    present in labels.items() if present > 0]

                #text = text_tokenizer(line['text'], **text_tokenizer_kwargs)
                text = line['text']
                # Ensure proper encoding
                try:
                    text = text.encode('latin-1').decode('utf8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    if self.verbose:
                        print("Removed sentence for bad encoding")
                    continue

                text = text_tokenizer(text, **text_tokenizer_kwargs)

                # If the tokenizer removes the text, carry on
                if text == None:
                    continue
                if isinstance(text, list):
                    text = ' '.join(text)

                # Ignore all remaining utf8 encodings and bring to 'plain' text
                text = text.encode('ascii', 'ignore').decode('ascii')

                # If more than 1 emotion is present, multiple examples are created
                if (not self.first_label_only):
                    for i, emotion in enumerate(present_emotions):
                        label = label_map[source][emotion]

                        datasets[source][split][label].append(
                            {'idx': id, 'labels': label, 'text': text})
                        source_lengths[source] = id + i + 1
                else:
                    label = label_map[source][present_emotions[0]]

                    datasets[source][split][label].append(
                        {'idx': id, 'labels': label, 'text': text})
                    source_lengths[source] = id + 1


        for source in datasets.keys():
            if len(datasets[source].keys()) == 1 and 'all' in datasets[source].keys():
                class_lengths = {k: len(datasets[source]['all'][k])
                                for k in datasets[source]['all'].keys()}
                for c, l in class_lengths.items():
                    train_l = int(self.split_ratio * l)
                    datasets[source]['train'][c] = datasets[source]['all'][c][:train_l]
                    datasets[source]['test'][c] = datasets[source]['all'][c][train_l:]

                del datasets[source]['all']

        self.datasets = datasets
        self.source_lengths = source_lengths
        self.label_map = label_map

        self.inv_label_map = {source: {val: key for key,
                                val in label_map[source].items()} for source in label_map.keys()}

        # Remove classes with limited data
        total_removed, total_data_removed = 0, 0
        removing = []
        for source in datasets.keys():
            n_classes = len(datasets[source]['train'].keys())
            for c in datasets[source]['train'].keys():
                train_size = len(datasets[source]['train'][c])
                test_size = len(datasets[source]['test'][c])

                keep = (train_size >= 96 and test_size >= 64)

                if (not keep):
                    if self.verbose:
                        print("Removed {:}/{:} for too little data |train|={}, |test|={}".
                            format(source, self.inv_label_map[source][c], train_size, test_size))
                    total_removed += 1
                    total_data_removed += train_size + test_size

                    self.source_lengths[source] -= train_size + test_size

                    removing.append((source, c))

        for source, c in removing:
            del datasets[source]['train'][c]
            del datasets[source]['test'][c]

        if self.verbose:
            print("Removed a total of {:} classes and {:} examples.".format(
                total_removed, total_data_removed))

        for source in datasets.keys():
            assert len(datasets[source]['train'].keys()) >= 2, print(
                f"{source} has too few classes remaining.")

    @property
    def lens(self):
        """Lengths of the individual datasets
        """
        return self.source_lengths

    def __getitem__(self, i):
        return self.datasets.get(i, None)

"""
    def get_dataloader(self, source_name, device, k=4, tokenizer=None, shuffle=True):
        Generates a dataloader from a specified dataset.
        See MetaStratifiedLoader for more.

        Args:
            source_name(str): a dataset from one of the processed ones.
            k(int, optional): the k-shot. Defaults to 4.
            tokenizer(callable, optional): function that processes list of strings to PyTorch tensor. Defaults to None.
            shuffle(boolean, optional): whether or not to shuffle the train data. Defaults to True.

        Returns:
            dataloaders: iterable of data_loaders. First is train, last is test.
        data_loaders = []
        for split in self.datasets[source_name].keys():
            source_dict = self.datasets[source_name]
            dataloader = MetaStratifiedLoader(source_dict=source_dict,
                                              split=split,
                                              class_to_int=self.label_map[source_name],
                                              k=k,
                                              tokenizer=tokenizer,
                                              shuffle=shuffle,
                                              device=device
                                              )

            if split == 'train':
                data_loaders.insert(0, dataloader)
            else:
                data_loaders.append(dataloader)

        return data_loaders

"""
