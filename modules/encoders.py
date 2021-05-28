import re

import torch
import torch.nn as nn
from transformers import AutoModel
from memory_profiler import profile

class TransformerEncoder(nn.Module):

    def __init__(self, name, nu):
        """BERT transformer for sequence classification.

        Args:
            name (str): name of pre-trained model
            config (dict): config file from Huggingface
            nu (int): min layer to freeze. Set to 11 for fully frozen, -1 for fully trainable
        """

        super().__init__()

        self.name = name
        self.nu = nu

        self.model = AutoModel.from_pretrained(name)

        for name, param in self.model.named_parameters():

            transformer_layer = re.search("(?:encoder\.layer\.)([0-9]+)", name)
            if transformer_layer and (int(transformer_layer.group(1)) > nu):
                param.requires_grad = True
            elif 'pooler' in name:
                param.requires_grad = False
            elif nu < 0:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.out_dim = self.model.config.hidden_size

    def forward(self, model_input):

        y = self.model(**model_input)['pooler_output']

        return y

class AWEEncoder(nn.Module):

    def __init__(self, vocab_length, out_dim=768, padding_idx=0, freeze=True):

        super().__init__()

        self.embedder = nn.Embedding(num_embeddings=vocab_length,
                                     embedding_dim=768,
                                     padding_idx=padding_idx)
        if freeze:
            self.embedder.requires_grad = False

        self.out_dim = out_dim

    def forward(self, model_input):

        return torch.mean(self.embedder(model_input['input_ids']), dim=1)
