import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class BertSequence(nn.Module):

    def __init__(self, name, config, nu):
        """BERT transformer for sequence classification.

        Args:
            name (str): name of pre-trained model
            config (dict): config file from Huggingface
            nu (int): min layer to freeze. Set to 11 for fully frozen, -1 for fully trainable
        """

        super().__init__()

        self.name = name
        self.config = config
        self.nu = nu

        self.model = BertModel.from_pretrained(self.name, config=self.config)

        for param in self.model.base_model.embeddings.parameters():
            param.requires_grad = False

        for name, param in self.model.base_model.encoder.layer.named_parameters():
            l = int(name[0])
            if l <= self.nu:
                param.requires_grad = False

    def forward(self, text, attn_mask=None):

        return self.model(text, attn_mask)['pooler_output']
