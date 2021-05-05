from collections import defaultdict

from datasets import load_dataset
from data.utils.data_loader import StratifiedLoader

class go_emotions():
    """Class for the 'GoEmotions Dataset'
    """

    def __init__(self, first_label_only=True):
        """
        Class for the 'GoEmotions Dataset'.

        Args:
            first_label_only (boolean, optional): boolean indicating whether to only use the first label when multiple labels are present. Defaults to True
        """

        self.dataset_name = 'go_emotions'
        self.first_label_only = first_label_only

    def prep(self, text_tokenizer=lambda x: x, text_tokenizer_kwargs=dict()):
        """
        Generates dataset from the Huggingface dataset.

        Args:
            text_tokenizer (callable, optional): function that processes a line of text. Defaults to identity (raw text).
        """

        dataset = load_dataset(self.dataset_name)
        train_set = dataset['train']
        dev_set = dataset['validation']
        test_set = dataset['test']

        # filter out al instances where the emotion is neutral
        train_set = train_set.filter(lambda example: not 27 in example['labels'])
        dev_set = dev_set.filter(lambda example: not 27 in example['labels'])
        test_set = test_set.filter(lambda example: not 27 in example['labels'])

        datasets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        #datasets['go_emotions'] = defaultdict()
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
        inv_label_map = {v: k for k, v in label_map.items()}


        for set in [('train', train_set), ('validation', dev_set), ('test', test_set)]:
            split_name = set[0]
            split_dataset = set[1]
            for instance in split_dataset:
                id = source_lengths.get('go_emotions', 0)

                text = text_tokenizer(instance['text'], **text_tokenizer_kwargs)
                if text == None:
                    continue
                if isinstance(text, list):
                    text = ' '.join(text)

                # check how to handle multiple labels
                labels = instance['labels']
                if (len(labels) > 1) and not self.first_label_only:
                    for label_idx, label in enumerate(labels):
                        #label = inv_label_map[label]
                        label = label
                        datasets['go_emotions'][split_name][label].append({'idx': (id + label_idx), 'labels': label, 'text': text})
                    source_lengths['go_emotions'] = id + len(labels)
                else:
                    #label = inv_label_map[labels[0]]
                    label = labels[0]
                    source_lengths['go_emotions'] = id + 1

                datasets['go_emotions'][split_name][label].append({'idx': id, 'labels': label, 'text': text})

        self.datasets = datasets
        self.source_lengths = source_lengths
        self.label_map = label_map

    @property
    def lens(self):
        """
        Lengths of the individual datasets
        """

        return self.source_lengths

    def get_dataloader(self, source_name,  k=4, tokenizer=None, shuffle=True):
        """
        Generates a dataloader from a specified dataset.
        See MetaStratifiedLoader for more.

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
            dataloader = MetaStratifiedLoader(source_dict=source_dict,
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

    def __getitem__(self, i):
        return self.datasets.get(i, None)

# DEBUG
if __name__ == "__main__":
    dataset = go_emotions()
    dataset.prep()
