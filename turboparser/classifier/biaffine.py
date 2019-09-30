# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import functional as F


class PairwiseBilinear(nn.Module):
    '''
    A bilinear module that deals with broadcasting for efficient memory usage.

    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)'''
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(
            torch.zeros(input1_size, input2_size, output_size))

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        batch_size = input1_size[0]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        output_dim = self.input2_size * self.output_size
        intermediate = torch.mm(input1.view(-1, input1_size[-1]),
                                self.weight.view(-1, output_dim))

        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)

        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        mid_dim = input1_size[1] * self.output_size
        output = intermediate.view(batch_size, mid_dim, input2_size[2])\
            .bmm(input2)

        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(batch_size, input1_size[1],
                             self.output_size, input2_size[1]).transpose(2, 3)

        return output

    def extra_repr(self):
        return 'in_features_1={}, in_features_2={}, out_features={}'.format(
            self.input1_size - 1, self.input2_size - 1, self.output_size)


class PairwiseBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.w = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

    def forward(self, input1, input2):
        # add bias
        bias1 = input1.new_ones(*input1.size()[:-1], 1)
        input1 = torch.cat([input1, bias1], len(input1.size()) - 1)
        bias2 = input2.new_ones(*input2.size()[:-1], 1)
        input2 = torch.cat([input2, bias2], len(input2.size()) - 1)

        return self.w(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size,
                 hidden_func=F.relu, dropout=0):
        super().__init__()
        self.output_size = output_size
        self.w1 = nn.Linear(input1_size, hidden_size)
        self.w2 = nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size,
                                             output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        hidden1 = self.hidden_func(self.w1(input1))
        hidden1 = self.dropout(hidden1)
        hidden2 = self.hidden_func(self.w2(input2))
        hidden2 = self.dropout(hidden2)
        return self.scorer(hidden1, hidden2)
