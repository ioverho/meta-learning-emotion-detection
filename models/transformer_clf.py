from models.seqtransformer import SeqTransformer
from modules.mlp_clf import SF_CLF
from torch import nn

class Transformer_CLF(nn.Module):
    """Transformer based sequence classiifer for finetuning baseline"""
    def __init__(self, config):
        super().__init__()
        self.encoder = SeqTransformer(config)
        self.classifer = SF_CLF(config['n_classes'], config['hidden_dims'])

    def forward(self, text, attn_mask=None):
        logits = self.classifer(self.encoder(text, attn_mask))
        return logits

    def encode(self, text, attn_mask=None):
        return self.encoder(text, attn_mask)