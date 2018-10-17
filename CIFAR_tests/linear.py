


'''

Linear module modified for the expander and clipping individual gradients.

This code is due to Mikko Heikkilä (@mixheikk)

'''


import math

import torch
from torch.nn.parameter import Parameter
import functional as F
from torch.nn.modules import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, batch_size = None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        if batch_size is not None:
            self.weight = Parameter(torch.Tensor(batch_size, out_features, in_features))
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            if batch_size is not None:
                self.bias = Parameter(torch.Tensor(batch_size, out_features))
            else:
                self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
