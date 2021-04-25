import torch
import torch.nn as nn

from modules.encoders import BertSequence
from modules.mlp_clf import MLP

class MetaBert(nn.Module):
    def __init__(self, config):
        """BERT sequence model for meta-learning.

        Args:
            bert_encoder ([type]): [description]
            mlp ([type]): [description]
            softmax_clf ([type]): [description]
        """
        super().__init__()

        self.pt_bert_name = config['bert_name']
        self.pt_bert_config = config['bert_config']
        self.nu = config['nu']

        self.encoder = BertSequence(name=self.pt_bert_name,
                                    config=self.pt_bert_config,
                                    nu=self.nu)

        self.hidden_dims = config['hidden_dims']
        self.act_fn = nn.ReLU if config['act_fn'] == 'ReLU' else nn.tanh

        self.mlp = MLP(encoder_output_size=self.encoder.config.hidden_size,
                       hidden_dims=self.hidden_dims,
                       act_fn=self.act_fn)


        self.out_dim = self.mlp.out_dim

    def forward(self, text, attn_mask=None):

        y = self.encoder(text, attn_mask)
        y = self.mlp(y)

        return y
