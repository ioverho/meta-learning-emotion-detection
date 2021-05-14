from data.unified_emotion_numpy import unified_emotion
from data.go_emotions_numpy import go_emotions
from data.utils.tokenizer import manual_tokenizer

class meta_dataset():
    def __init__(self, unified_file_path="./data/datasets/unified-dataset.jsonl",
                 include=['crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                          'emotion-cause', 'grounded_emotions', 'ssec', 'tec', 'go_emotions'],
                 split_ratio=0.8, verbose=False, first_label_only=False):

        self.unified = unified_emotion(unified_file_path,
                                       include=include,
                                       split_ratio=split_ratio,
                                       verbose=verbose,
                                       first_label_only=first_label_only)

        if 'go_emotions' in include:
            self.go_emotions = go_emotions(first_label_only=first_label_only,
                                           verbose=verbose)
        else:
            self.go_emotions = None

    def prep(self, text_tokenizer=lambda x: x, text_tokenizer_kwargs=dict()):

        self.unified.prep(text_tokenizer=manual_tokenizer,
                          text_tokenizer_kwargs=text_tokenizer_kwargs)

        if self.go_emotions != None:
            self.go_emotions.prep(text_tokenizer=manual_tokenizer,
                                  text_tokenizer_kwargs=text_tokenizer_kwargs)

            self.datasets = {**dict(self.unified.datasets),
                             **dict(self.go_emotions.datasets)}
            self.label_map = {**dict(self.unified.label_map),
                              **{'go_emotions': dict(self.go_emotions.label_map)}}
            self.inv_label_map = {**dict(self.unified.inv_label_map),
                                  **{'go_emotions': dict(self.go_emotions.inv_label_map)}}
            self.source_lengths = {**self.unified.source_lengths,
                                   **self.go_emotions.source_lengths}
        else:
            self.datasets = dict(self.unified.datasets)
            self.label_map = dict(self.unified.label_map)
            self.inv_label_map = dict(self.unified.inv_label_map)
            self.source_lengths = dict(self.unified.source_lengths)

    @property
    def lens(self):
        """Lengths of the individual datasets
        """
        return self.source_lengths

    def __getitem__(self, i):
        return self.datasets.get(i, None)
