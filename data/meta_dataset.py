from data.unified_emotion_numpy import unified_emotion
from data.go_emotions_numpy import go_emotions
from data.utils.tokenizer import manual_tokenizer

def info():
    return [{'source': 'affectivetext', 'size': 250, 'domain': 'headlines', 'classes': 6, 'special': 'non-discrete, multiple labels'},
            {'source': 'crowdflower', 'size': 40000, 'domain': 'tweets', 'classes': 14, 'special': 'includes no-emotions class'},
            {'source': 'dailydialog', 'size': 13000, 'domain': 'conversations', 'classes': 6, 'special': 'includes no-emotions class'},
            {'source': 'electoraltweets', 'size': 4058, 'domain': 'tweets', 'classes': 8, 'special': 'includes no-emotions class'},
            {'source': 'emobank', 'size': 10000, 'domain': 'headlines', 'classes': 3, 'special': 'VAD regression'},
            {'source': 'emoint', 'size': 7097, 'domain': 'tweets', 'classes': 6, 'special': 'annotated by experts'},
            {'source': 'emotion-cause', 'size': 2414, 'domain': 'artificial', 'classes': 6, 'special': 'N/A'},
            {'source': 'fb-valence-arousal-anon', 'size': 2800, 'domain': 'facebook', 'classes': 3, 'special': 'VA regression'},
            {'source': 'go_emotions', 'size': 58000, 'domain': 'reddit posts', 'classes': 27, 'special': 'N/A'},
            {'source': 'grounded_emotions', 'size': 2500, 'domain': 'tweets', 'classes': 2, 'special': 'N/A'},
            {'source': 'ssec', 'size': 4868, 'domain': 'tweets', 'classes': 8, 'special': 'multiple labels per sentence'},
            {'source': 'tales-emotion', 'size': 15302, 'domain': 'fairytales', 'classes': 6, 'special': 'includes no-emotions class'},
            {'source': 'tec', 'size': 21051, 'domain': 'tweets', 'classes': 7, 'special': 'annotated by experts'}
            ]

task_label_dict = {
    'go_emotions': 27,
    'crowdflower': 8,
    'dailydialog': 7,
    'electoraltweets': 10,
    'emoint': 4,
    'emotion-cause': 6,
    'grounded_emotions': 2,
    'ssec': 7,
    'tales-emotion': 7,
    'tec': 6,
}

class meta_dataset():
    def __init__(self, unified_file_path="./data/datasets/unified-dataset.jsonl",
                 include=['crowdflower', 'dailydialog', 'electoraltweets', 'emoint',
                          'emotion-cause', 'grounded_emotions', 'ssec', 'tec', 'go_emotions'],
                 split_ratio=[0.7, 0.15, 0.15], verbose=False, first_label_only=False):

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
