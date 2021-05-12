from models.seqtransformer import SeqTransformer
from modules.mlp_clf import SF_CLF
from torch import nn

class Transformer_CLF(nn.Module):
    """Transformer based sequence classiifer for finetuning baseline"""
    def __init__(self, args):
        super().__init__()
        self.encoder = SeqTransformer(args)
        self.classifer = SF_CLF(args.num_classes, args.hidden_dims)

    def forward(self, model_input):
        x = self.encoder(model_input)
        logits = self.classifer(x)
        return logits
