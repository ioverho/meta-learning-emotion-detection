import numpy as np
import torch


class UnifiedMetaStratifiedLoader():
    def __init__(self, source_dict, split, class_to_int, k, tokenizer):

        self.k = k
        self.split = split

        self.class_to_int = class_to_int
        self.int_to_class = {v: k for k, v in class_to_int.items()}

        self.data = source_dict[split]

        self.labels = list(self.data.keys())
        self._len = sum([len(self.data[k]) for k in self.labels])

        self.tokenizer = tokenizer

    def __len__(self):
        return self._len

    def __next__(self):

        text = []
        labels = []
        for c in self.data.keys():
            samples = np.random.choice(self.data[c], self.k)
            text.extend([s['text'] for s in samples])
            labels.extend([s['labels'] for s in samples])

        if self.tokenizer == None:
            return labels, text
        else:
            encoded = self.tokenizer(text, padding=True, return_tensors="pt")
            text = encoded['input_ids'].T
            mask = encoded['attention_mask'].T

            labels = torch.LongTensor(labels)

            return labels, text, mask
