# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import math

import torch
from torch import nn
from torch.nn import functional as F


class ImputationLayer(nn.Module):
    def __init__(self, d_in, bias=True):
        super(ImputationLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(d_in, d_in))
        if bias:
            self.b = nn.Parameter(torch.Tensor(d_in))
        else:
            self.register_buffer('b', None)
        mask = 1. - torch.eye(d_in)
        self.register_buffer('mask', mask)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # batch, features
        return F.linear(x, self.mask * self.W, self.b)