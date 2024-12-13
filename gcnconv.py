import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module


# define a single graph convolution layer
class GCNConv(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # self.weight = Parameter(torch.eye(in_features) * c_max)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            # self.register_parameter('bias', None)
            pass
        self.reset_parameters()

    # initializing weights 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # (X, W)
        support = torch.matmul(input, self.weight) 
        # (A, XW)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SGConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_layers, bias=True):
        super(SGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        
        # self.weight = Parameter(torch.eye(in_features) * c_max)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            # self.register_parameter('bias', None)
            pass
        self.reset_parameters()

    # initializing weights 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # (X, W)
        support = torch.matmul(input, self.weight) 

        A_prev = torch.eye(adj.shape[0]).to(adj.device)
        for _ in range(self.num_layers):
            A = torch.matmul(A_prev, adj)
            A_prev = A

        # (A^k, XW)
        output = torch.matmul(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

