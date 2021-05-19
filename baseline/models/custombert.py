import torch
import torch.nn as nn

from modules.encoders import TransformerEncoder
from modules.mlp_clf import MLP

class CustomBERT(nn.Module):
    def __init__(self, num_classes):
        """Custom BERT Transformer based sequence model.
        """
        super().__init__()

        # BERT encoder
        self.encoder = TransformerEncoder(name='bert-base-uncased', nu=-1)

        # MLP
        self.act_fn = nn.ReLU
        self.mlp = MLP(encoder_output_size=self.encoder.out_dim,
                       hidden_dims=[128],
                       act_fn=self.act_fn)

        # final classification layer
        self.num_classes = num_classes
        self.clf = nn.Linear(in_features=self.mlp.out_dim, out_features=self.num_classes)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def replace_clf(self, num_classes):
        # replace the clf with a new final clf which has new number of classes
        self.num_classes = num_classes
        self.clf = nn.Linear(in_features=self.mlp.out_dim, out_features=self.num_classes)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # create the predictions
        y = self.encoder(input_ids, attention_mask, token_type_ids=token_type_ids)
        y = self.mlp(y)
        y = self.clf(y)

        # calculate the loss
        loss = self.loss_fn(y, labels)

        # return the predictions and the loss
        return y, loss
