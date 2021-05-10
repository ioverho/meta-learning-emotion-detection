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
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, model_input):

        y = self.model(**model_input)['pooler_output']

        return y
