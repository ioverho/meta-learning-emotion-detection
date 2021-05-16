import torch
import torch.nn as nn

from modules.encoders import TransformerEncoder, AWEEncoder
from modules.mlp_clf import MLP

class SeqTransformer(nn.Module):
    def __init__(self, args):
        """Transformer based sequence model for meta-learning. Only encodes, does not classify.

        Args:
            bert_encoder ([type]): [description]
            mlp ([type]): [description]
            softmax_clf ([type]): [description]
        """
        super().__init__()

        if args['encoder_name'].lower() == 'random':
            self.encoder = AWEEncoder(args['vocab_length'], out_dim=768,
                                      padding_idx=0, freeze=True)
        else:
            self.pt_encoder = args['encoder_name']
            self.nu = args['nu']

            self.encoder = TransformerEncoder(name=self.pt_encoder,
                                              nu=self.nu)

        self.hidden_dims = args['hidden_dims']
        self.act_fn = nn.ReLU if args['act_fn'] == 'ReLU' else nn.Tanh

        self.mlp = MLP(encoder_output_size=self.encoder.out_dim,
                       hidden_dims=self.hidden_dims,
                       act_fn=self.act_fn)

        self.out_dim = self.mlp.out_dim

    def forward(self, model_input):

        y = self.encoder(model_input)
        y = self.mlp(y)

        return y
