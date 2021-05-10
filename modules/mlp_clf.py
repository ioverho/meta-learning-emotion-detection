import torch
import torch.nn as nn
import torch.functional as F
from memory_profiler import profile

class MLP(nn.Module):

    def __init__(self, encoder_output_size, hidden_dims, act_fn=nn.ReLU):
        """MLP module.

        Args:
            encoder_output_size (int): output size of encoder. Input should be tensor of size [batch_size, encoder_output_size]
            n_classes (int): number of classes for output classification
            hidden_dims (iterable of ints): hidden dimensions to which the MLP projects to
            act_fn (torch.nn, optional): non-linear activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        mlp_layers = []
        for h_in, h_out in zip([encoder_output_size] + hidden_dims[:-1], hidden_dims):
            mlp_layers.append(nn.Linear(in_features=h_in, out_features=h_out))
            mlp_layers.append(act_fn())
        self.mlp = nn.Sequential(*mlp_layers)

        self.out_dim = hidden_dims[-1]

    #@profile
    def forward(self, x):

        y = self.mlp(x)

        return y

class SF_CLF(nn.Module):

    def __init__(self, n_classes, hidden_dims):
        """Softmax classification module.

        Args:
            encoder_output_size (int): output size of encoder. Input should be tensor of size [batch_size, encoder_output_size]
            n_classes (int): number of classes for output classification
            hidden_dims (iterable of ints): hidden dimensions to which the MLP projects to
            act_fn (torch.nn, optional): non-linear activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.clf = nn.Linear(in_features=hidden_dims[-1], out_features=n_classes)

    def forward(self, x):

        logits = self.clf(x)

        return logits
