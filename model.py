import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx
from scipy.sparse import csr_matrix

from torch_geometric.nn import GCNConv, SGConv, GCN2Conv, GINConv, ChebConv, MixHopConv, SAGEConv, FAConv
# from gcnconv import GCNConv, SGConv
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, from_scipy_sparse_matrix, to_undirected, get_laplacian, to_dense_adj
# from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp


# GCN
class GCNModel(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
        super(GCNModel, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.mlp_layers = mlp_layers
        self.alpha = alpha 
        self.gamma = gamma
        print("Using GCN model")
        
        # self.init_layer = nn.Linear(input_dim, hidden_dim)
      
        for i in range(self.num_layers):
            if i == 0:
                self.gcn_convs.append(GCNConv(self.dataset.num_features, self.hidden_dim).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))
            elif i == self.num_layers - 1:
                self.gcn_convs.append(GCNConv(self.hidden_dim, self.dataset.num_classes).to(self.device))
            else:
                self.gcn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))


    def forward(self, x, edge_index):
        x = x.to(self.device)
        num_nodes = x.shape[0]
        
        # x_0 = x
        # x = F.dropout(x, p=self.dropout, training=True)
        # x = self.init_layer(x)
    
        '''
        adding self-loops and parallel edges in the graph
        '''
        # print(adj_matrix)
        # adj_matrix = adj_matrix.to(self.device)
        # adj_matrix = (self.gamma * adj_matrix)
        # adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
        # print(adj_matrix)
        # norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
        # print(norm_updated_adj_matrix)

        # edge_indices = torch.where(adj_matrix != 0)
        # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
        # adj_matrix_flatten = adj_matrix.reshape(num_nodes * num_nodes)
        # edge_weights = adj_matrix_flatten[adj_matrix_flatten != 0]

        # num_edges = edge_index.shape[1]
        # degrees = degree(edge_index[0])
        # print("before ", sum(degrees))
        
        # sp_adj_matrix = csr_matrix(adj_matrix)
        # edge_index = from_scipy_sparse_matrix(sp_adj_matrix)[0]
        # edge_index = edge_index.to(self.device)
        
        '''
        add parallel edges first then add self-loops
        '''

        edge_index = to_undirected(edge_index)
        updated_edge_index = remove_self_loops(edge_index)[0]
        num_edges = updated_edge_index.shape[1]
        if int(self.alpha) > 1:
            updated_edge_index = add_self_loops(updated_edge_index)[0]
        edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
        edge_weights_sl = torch.ones(num_nodes).to(self.device) * self.alpha
        if int(self.alpha) > 1:
            edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)
        else:
            edge_weights = edge_weights_pe

        # print("edge weights ", edge_weights.shape, "  ", updated_edge_index.shape)

      
        # message propagation through hidden layers
        for i in range(self.num_layers):
            x = self.gcn_convs[i](x, updated_edge_index, edge_weights)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x
    
    # normalization of the adjacency matrix
    # def normalize_adj(self, adj_matrix):
    #     # adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0]).to(self.device)
    #     num_neigh = adj_matrix.sum(dim = 1, keepdim = True)
    #     num_neigh = num_neigh.squeeze(1)
    #     # print(len(num_neigh[num_neigh == 0.0]))
    #     num_neigh = torch.sqrt(1 / num_neigh)
    #     degree_matrix = torch.diag(num_neigh)
    #     adj_matrix = torch.mm(degree_matrix, adj_matrix)
    #     adj_matrix = torch.mm(adj_matrix, degree_matrix)
    #     return adj_matrix



# FAConv
class FAModel(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
        super(FAModel, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.mlp_layers = mlp_layers
        self.alpha = alpha 
        self.gamma = gamma
        print("Using FAConv model")
        
        self.init_layer = nn.Linear(input_dim, self.hidden_dim)
        self.last_layer = nn.Linear(self.hidden_dim, self.dataset.num_classes)
      
        for i in range(self.num_layers):      
            self.gcn_convs.append(FAConv(self.hidden_dim, eps = 0.1, dropout = self.dropout, add_self_loops=False, normalize=False).to(self.device))
            self.lns.append(nn.LayerNorm(hidden_dim))


    def forward(self, x, edge_index):
        x = x.to(self.device)
        num_nodes = x.shape[0]
        
        x = self.init_layer(x)
        x_0 = x
        
        # x_0 = x
        # x = F.dropout(x, p=self.dropout, training=True)
        # x = self.init_layer(x)
        
        '''
        add parallel edges first then add self-loops
        '''

        # edge_index = to_undirected(edge_index)
        # edge_index = remove_self_loops(edge_index)[0]
        # self_loops_indices = torch.tensor([i for i in range(num_nodes)]).to(self.device)
        # self_loops_edge_index = torch.stack([self_loops_indices, self_loops_indices])
        # self_loops_edge_index = self_loops_edge_index.repeat(1, int(self.alpha)-1)
        # parallel_edge_index = edge_index.repeat(1, int(self.gamma))
        # updated_edge_index = torch.cat([parallel_edge_index, self_loops_edge_index], dim=1)

        # edge_index = to_undirected(edge_index)
        updated_edge_index = remove_self_loops(edge_index)[0]
        num_edges = updated_edge_index.shape[1]
        updated_edge_index = add_self_loops(updated_edge_index, num_nodes=num_nodes)[0]
        edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
        edge_weights_sl = torch.ones(num_nodes).to(self.device) * (self.alpha)
        edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)

        # print("edge weights ", edge_weights.shape, "  ", updated_edge_index.shape)

        # message propagation through hidden layers
        for i in range(self.num_layers):
            x = self.gcn_convs[i](x, x_0, updated_edge_index, edge_weight=edge_weights)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                  
        x = self.last_layer(x)
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x



# High frequency signal 
class HPFModel(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
        super(HPFModel, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.mlp_layers = mlp_layers
        self.alpha = alpha 
        self.gamma = gamma
        self.eps = 0.4
        print("Using High Pass Filter model")
        
        # self.init_layer = nn.Linear(input_dim, hidden_dim)
      
        for i in range(self.num_layers):
            if i == 0:
                self.gcn_convs.append(GCNConv(self.dataset.num_features, self.hidden_dim, normalize=False).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))
            elif i == self.num_layers - 1:
                self.gcn_convs.append(GCNConv(self.hidden_dim, self.dataset.num_classes,  normalize=False).to(self.device))
            else:
                self.gcn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim, normalize=False).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))


    def forward(self, x, edge_index):
        x = x.to(self.device)
        num_nodes = x.shape[0]
        
        # x_0 = x
        # x = F.dropout(x, p=self.dropout, training=True)
        # x = self.init_layer(x)
    
        '''
        adding self-loops and parallel edges in the graph
        '''
        
        # F_h = (e - 1)I + L
        
        # print(adj_matrix)
        adj_matrix = to_dense_adj(edge_index)[0]
        adj_matrix = adj_matrix.to(self.device)
        adj_matrix = (self.gamma * adj_matrix)
        adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
        # print(adj_matrix)
        norm_laplacian = self.normalize_laplacian(adj_matrix)
        updated_laplacian = (self.eps - 1) * torch.eye(num_nodes).to(self.device) + norm_laplacian
        src, tgt = torch.where(updated_laplacian > 0)
        updated_edge_index = torch.stack([src, tgt])
        updated_laplacian_flatten = updated_laplacian.reshape(num_nodes * num_nodes)
        updated_edge_weights = updated_laplacian_flatten[updated_laplacian_flatten > 0]
        # print(norm_updated_adj_matrix)

        # edge_indices = torch.where(adj_matrix != 0)
        # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
        # adj_matrix_flatten = adj_matrix.reshape(num_nodes * num_nodes)
        # edge_weights = adj_matrix_flatten[adj_matrix_flatten != 0]

        # num_edges = edge_index.shape[1]
        # degrees = degree(edge_index[0])
        # print("before ", sum(degrees))
        
        # sp_adj_matrix = csr_matrix(adj_matrix)
        # edge_index = from_scipy_sparse_matrix(sp_adj_matrix)[0]
        # edge_index = edge_index.to(self.device)
        
        '''
        add parallel edges first then add self-loops
        '''

        # edge_index = to_undirected(edge_index)
        # updated_edge_index = remove_self_loops(edge_index)[0]
        # num_edges = updated_edge_index.shape[1]
        # if int(self.alpha) > 1:
        #     updated_edge_index = add_self_loops(updated_edge_index)[0]
        # edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
        # edge_weights_sl = torch.ones(num_nodes).to(self.device) * self.alpha
        # if int(self.alpha) > 1:
        #     edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)
        # else:
        #     edge_weights = edge_weights_pe

        # print("edge weights ", edge_weights.shape, "  ", updated_edge_index.shape)
        
        # updated_laplacian, updated_weights = get_laplacian(updated_edge_index, edge_weights=None, normalization='sym', num_nodes=num_nodes)
        # new_updated_laplacian = add_self_loops(updated_laplacian)[0]
        # updated_weights = torch.cat([updated_weights, torch.ones(new_updated_laplacian.shape[1]) * (self.eps - 1)], dim = 0)
        # print(updated_laplacian.shape, "\t", updated_weights.shape)

        # message propagation through hidden layers
        for i in range(self.num_layers):
            x = self.gcn_convs[i](x, updated_edge_index, updated_edge_weights)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x
    
    # normalization of the adjacency matrix
    def normalize_laplacian(self, adj_matrix):
        # adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0]).to(self.device)
        num_neigh = adj_matrix.sum(dim = 1, keepdim = True)
        num_neigh = num_neigh.squeeze(1)
        # print(len(num_neigh[num_neigh == 0.0]))
        num_neigh = torch.sqrt(1 / num_neigh)
        degree_matrix = torch.diag(num_neigh)
        adj_matrix = torch.mm(degree_matrix, adj_matrix)
        adj_matrix = torch.mm(adj_matrix, degree_matrix)
        norm_laplacian = torch.eye(adj_matrix.shape[0]).to(adj_matrix.device) - adj_matrix
        return norm_laplacian




# SAGE
# class SAGEModel(nn.Module):

#     def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
#         super(SAGEModel, self).__init__()

#         self.num_layers = num_layers
#         self.dataset = dataset
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.device = device
#         self.sage_convs = nn.ModuleList()
#         self.lns = nn.ModuleList()
#         self.mlp_layers = mlp_layers
#         self.alpha = alpha 
#         self.gamma = gamma
#         print("Using SAGE model")
        
#         # self.init_layer = nn.Linear(input_dim, hidden_dim)
      
#         for i in range(self.num_layers):
#             if i == 0:
#                 self.sage_convs.append(SAGEConv(self.dataset.num_features, self.hidden_dim).to(self.device))
#                 self.lns.append(nn.LayerNorm(hidden_dim))
#             elif i == self.num_layers - 1:
#                 self.sage_convs.append(SAGEConv(self.hidden_dim, self.dataset.num_classes).to(self.device))
#             else:
#                 self.sage_convs.append(SAGEConv(self.hidden_dim, self.hidden_dim).to(self.device))
#                 self.lns.append(nn.LayerNorm(hidden_dim))


#     def forward(self, x, edge_index):
#         x = x.to(self.device)
#         num_nodes = x.shape[0]
        
#         # x_0 = x
#         # x = F.dropout(x, p=self.dropout, training=True)
#         # x = self.init_layer(x)
    
#         '''
#         adding self-loops and parallel edges in the graph
#         '''
#         # print(adj_matrix)
#         # adj_matrix = adj_matrix.to(self.device)
#         # adj_matrix = (self.gamma * adj_matrix)
#         # adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
#         # print(adj_matrix)
#         # norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
#         # print(norm_updated_adj_matrix)

#         # edge_indices = torch.where(adj_matrix != 0)
#         # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
#         # adj_matrix_flatten = adj_matrix.reshape(num_nodes * num_nodes)
#         # edge_weights = adj_matrix_flatten[adj_matrix_flatten != 0]

#         # num_edges = edge_index.shape[1]
#         # degrees = degree(edge_index[0])
#         # print("before ", sum(degrees))
        
#         # sp_adj_matrix = csr_matrix(adj_matrix)
#         # edge_index = from_scipy_sparse_matrix(sp_adj_matrix)[0]
#         # edge_index = edge_index.to(self.device)
        
#         '''
#         add parallel edges first then add self-loops
#         '''

#         # edge_index = to_undirected(edge_index)
#         # updated_edge_index = remove_self_loops(edge_index)[0]
#         # num_edges = updated_edge_index.shape[1]
#         # if int(self.alpha) > 1:
#         #     updated_edge_index = add_self_loops(updated_edge_index)[0]
#         # edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
#         # edge_weights_sl = torch.ones(num_nodes).to(self.device) * self.alpha
#         # if int(self.alpha) > 1:
#         #     edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)
#         # else:
#         #     edge_weights = edge_weights_pe

#         edge_index = to_undirected(edge_index)
#         edge_index = remove_self_loops(edge_index)[0]
#         self_loops_indices = torch.tensor([i for i in range(num_nodes)]).to(self.device)
#         self_loops_edge_index = torch.stack([self_loops_indices, self_loops_indices])
#         self_loops_edge_index = self_loops_edge_index.repeat(1, int(self.alpha)-1)
#         parallel_edge_index = edge_index.repeat(1, int(self.gamma))
#         updated_edge_index = torch.cat([parallel_edge_index, self_loops_edge_index], dim=1)

#         # print("edge weights ", edge_weights.shape, "  ", updated_edge_index.shape)

      
#         # message propagation through hidden layers
#         for i in range(self.num_layers):
#             x = self.sage_convs[i](x, updated_edge_index)
#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = self.lns[i](x)
#                 x = F.dropout(x, p=self.dropout, training=True)
                  
#         embedding = x
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x



# GIN
# class GINModel(nn.Module):
    
#     def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
#         super(GINModel, self).__init__()

#         self.num_layers = num_layers
#         self.dataset = dataset
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.device = device
#         self.gcn_convs = nn.ModuleList()
#         self.lns = nn.ModuleList()
#         self.mlp_layers = mlp_layers
#         self.alpha = alpha 
#         self.gamma = gamma
#         print("Using GIN model")
        
#         self.mlp_layer = nn.Sequential(
#                                     nn.Linear(self.dataset.num_features, self.hidden_dim),
#                                     nn.ReLU(),
#                                     nn.Linear(self.hidden_dim, self.dataset.num_classes))
      
#         self.gin_conv = GINConv(self.mlp_layer, eps = 0.1, train_eps = False)


#     def forward(self, x, edge_index):
#         x = x.to(self.device)
#         num_nodes = x.shape[0]
#         # x_0 = x
#         # x = F.dropout(x, p=self.dropout, training=True)
#         # x = self.init_layer(x)
    
    
#         # adj_matrix = adj_matrix.to(self.device)
#         # adj_matrix = (self.gamma * adj_matrix)
#         # adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
#         # norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
#         # print(norm_updated_adj_matrix)

#         # edge_indices = torch.where(adj_matrix != 0)
#         # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
#         # degrees = degree(edge_index[0])
#         # print("before ", sum(degrees))
#         # sp_adj_matrix = csr_matrix(adj_matrix)
#         # edge_index = from_scipy_sparse_matrix(sp_adj_matrix)[0]
#         # edge_index = edge_index.to(self.device)
        
        
#         '''
#         add parallel edges first then add self-loops
#         '''

        
#         # edge_index = to_undirected(edge_index)
#         # updated_edge_index = remove_self_loops(edge_index)[0]
#         # num_edges = updated_edge_index.shape[1]
#         # updated_edge_index = add_self_loops(updated_edge_index)[0]

#         edge_index = to_undirected(edge_index)
#         edge_index = remove_self_loops(edge_index)[0]
#         self_loops_indices = torch.tensor([i for i in range(num_nodes)]).to(self.device)
#         self_loops_edge_index = torch.stack([self_loops_indices, self_loops_indices])
#         self_loops_edge_index = self_loops_edge_index.repeat(1, int(self.alpha)-1)
#         parallel_edge_index = edge_index.repeat(1, int(self.gamma))
#         updated_edge_index = torch.cat([parallel_edge_index, self_loops_edge_index], dim=1)
        
#         # print(updated_edge_index.shape, "   ", edge_weights.shape)
      
#         # message propagation through hidden layers 
#         x = self.gin_conv(x, updated_edge_index)    
#         x = F.relu(x)
#         # x = self.lns(x)
#         x = F.dropout(x, p=self.dropout, training=True)
            
#         embedding = x
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x



# SGC
# class SGCModel(nn.Module):

#     def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
#         super(SGCModel, self).__init__()

#         self.num_layers = num_layers
#         self.dataset = dataset
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.device = device
#         self.gcn_convs = nn.ModuleList()
#         self.lns = nn.ModuleList()
#         self.mlp_layers = mlp_layers
#         self.alpha = alpha 
#         self.gamma = gamma
#         print("Using SGC model")
      
#         self.last_layer = nn.Linear(hidden_dim, self.dataset.num_classes)
      
#         self.sgc_model = SGConv(self.dataset.num_features, self.hidden_dim, self.num_layers).to(self.device)
#         self.lns = nn.LayerNorm(hidden_dim)


#     def forward(self, x, edge_index):
#         x = x.to(self.device)
#         num_nodes = x.shape[0]
#         # x_0 = x
#         # x = F.dropout(x, p=self.dropout, training=True)
#         # x = self.init_layer(x)
        
#         # adj_matrix = adj_matrix.to(self.device)

#         # adj_matrix = (self.gamma * adj_matrix)
#         # adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
#         # norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
#         # print(norm_updated_adj_matrix)

#         '''
#         adjacency matrix to edge index conversion
#         '''
#         # edge_indices = torch.where(adj_matrix != 0)
#         # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
#         # edge_index = from_scipy_sparse_matrix(adj_matrix)
#         # print("edge index ", edge_index.shape)
        
#         '''
#         add parallel edges first then add self-loops
#         '''

#         # edge_index = remove_self_loops(edge_index)[0]
#         # self_loops_indices = torch.tensor([i for i in range(num_nodes)]).to(self.device)
#         # self_loops_edge_index = torch.stack([self_loops_indices, self_loops_indices])
#         # self_loops_edge_index = self_loops_edge_index.repeat(1, int(self.alpha)-1)
#         # parallel_edge_index = edge_index.repeat(1, int(self.gamma))
#         # updated_edge_index = torch.cat([parallel_edge_index, self_loops_edge_index], dim=1)
        
#         # edge_index = to_undirected(edge_index)
#         updated_edge_index = remove_self_loops(edge_index)[0]
#         num_edges = updated_edge_index.shape[1]
#         if int(self.alpha) > 1:
#             updated_edge_index = add_self_loops(updated_edge_index)[0]
#         edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
#         edge_weights_sl = torch.ones(num_nodes).to(self.device) * self.alpha
#         if int(self.alpha) > 1:
#             edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)
#         else:
#             edge_weights = edge_weights_pe
       
#         x = self.sgc_model(x, updated_edge_index, edge_weights)
#         x = self.lns(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=True)
#         x = self.last_layer(x)
          
#         embedding = x
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x

#     # normalization of the adjacency matrix
#     # def normalize_adj(self, adj_matrix):
#     #     # adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0]).to(self.device)
#     #     num_neigh = adj_matrix.sum(dim = 1, keepdim = True)
#     #     num_neigh = num_neigh.squeeze(1)
#     #     # print(len(num_neigh[num_neigh == 0.0]))
#     #     num_neigh = torch.sqrt(1 / num_neigh)
#     #     degree_matrix = torch.diag(num_neigh)
#     #     adj_matrix = torch.mm(degree_matrix, adj_matrix)
#     #     adj_matrix = torch.mm(adj_matrix, degree_matrix)
#     #     return adj_matrix



# MixHop
# class MixHopModel(nn.Module):

#     def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
#         super(MixHopModel, self).__init__()

#         self.num_layers = num_layers
#         self.dataset = dataset
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.device = device
#         self.gcn_convs = nn.ModuleList()
#         self.lns = nn.ModuleList()
#         self.mlp_layers = mlp_layers
#         self.alpha = alpha 
#         self.gamma = gamma
#         print("Using MixHop model")
        
#         self.last_layer = nn.Linear(3*hidden_dim, self.dataset.num_classes)
      
#         self.mixhop_conv = MixHopConv(self.dataset.num_features, self.hidden_dim, powers=[0, 1, 2], add_self_loops=False).to(self.device)
#         self.lns = nn.LayerNorm(3*hidden_dim)
    

#     def forward(self, x, edge_index):
#         x = x.to(self.device)
#         num_nodes = x.shape[0]
        
#         # x_0 = x
#         # x = F.dropout(x, p=self.dropout, training=True)
#         # x = self.init_layer(x)
    
#         '''
#         adding self-loops and parallel edges in the graph
#         '''
#         # print(adj_matrix)
#         # adj_matrix = adj_matrix.to(self.device)
#         # adj_matrix = (self.gamma * adj_matrix)
#         # adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
#         # print(adj_matrix)
#         # norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
#         # print(norm_updated_adj_matrix)

#         # edge_indices = torch.where(adj_matrix != 0)
#         # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
#         # adj_matrix_flatten = adj_matrix.reshape(num_nodes * num_nodes)
#         # edge_weights = adj_matrix_flatten[adj_matrix_flatten != 0]

#         # num_edges = edge_index.shape[1]
#         # degrees = degree(edge_index[0])
#         # print("before ", sum(degrees))
        
#         # sp_adj_matrix = csr_matrix(adj_matrix)
#         # edge_index = from_scipy_sparse_matrix(sp_adj_matrix)[0]
#         # edge_index = edge_index.to(self.device)
        
#         '''
#         add parallel edges first then add self-loops
#         '''

#         edge_index = to_undirected(edge_index)
#         updated_edge_index = remove_self_loops(edge_index)[0]
#         num_edges = updated_edge_index.shape[1]
#         updated_edge_index = add_self_loops(updated_edge_index)[0]
#         edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
#         edge_weights_sl = torch.ones(num_nodes).to(self.device) * self.alpha
#         edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)
        
#         # print("edge weights ", edge_weights.shape, "  ", updated_edge_index.shape)

      
#         # message propagation through hidden layers
#         x = self.mixhop_conv(x, updated_edge_index, edge_weights)
#         x = self.lns(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=True)
#         x = self.last_layer(x)
                  
#         embedding = x
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x





# ChebConv
# class ChebConvModel(nn.Module):

#     def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
#         super(ChebConvModel, self).__init__()

#         self.num_layers = num_layers
#         self.dataset = dataset
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.device = device
#         self.gcn_convs = nn.ModuleList()
#         self.lns = nn.ModuleList()
#         self.mlp_layers = mlp_layers
#         self.alpha = alpha 
#         self.gamma = gamma
#         print("Using ChebConv model")
      
#         self.cheb_conv = ChebConv(self.dataset.num_features, self.hidden_dim, K = self.num_layers, normalization = 'sym')
#         self.last_layer = nn.Linear(self.hidden_dim, self.dataset.num_classes)
#         self.lns = nn.LayerNorm(self.hidden_dim)


#     def forward(self, x, edge_index):
#         x = x.to(self.device)
#         num_nodes = x.shape[0]
#         # x_0 = x
#         # x = F.dropout(x, p=self.dropout, training=True)
#         # x = self.init_layer(x)
    
#         '''
#         adding self-loops and parallel edges in the graph
#         '''

#         # adj_matrix = adj_matrix.to(self.device)
#         # adj_matrix = (self.gamma * adj_matrix)
#         # adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
#         # norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
#         # print(norm_updated_adj_matrix)

#         # edge_indices = torch.where(adj_matrix != 0)
#         # edge_index = torch.stack([edge_indices[0], edge_indices[1]])
#         # degrees = degree(edge_index[0])
#         # print("before ", sum(degrees))
#         # sp_adj_matrix = csr_matrix(adj_matrix)
#         # edge_index = from_scipy_sparse_matrix(sp_adj_matrix)[0]
#         # edge_index = edge_index.to(self.device)
        
        
#         '''
#         add parallel edges first then add self-loops
#         '''

#         edge_index = to_undirected(edge_index)
#         updated_edge_index = remove_self_loops(edge_index)[0]
#         num_edges = updated_edge_index.shape[1]
#         updated_edge_index = add_self_loops(updated_edge_index)[0]
#         edge_weights_pe = torch.ones(num_edges).to(self.device) * self.gamma
#         edge_weights_sl = torch.ones(num_nodes).to(self.device) * self.alpha
#         edge_weights = torch.cat([edge_weights_pe, edge_weights_sl], dim = 0)
       
#         # print(updated_edge_index.shape, "   ", edge_weights.shape)
        
#         # message propagation through hidden layers 
#         x = self.cheb_conv(x, updated_edge_index)    
#         x = F.relu(x)
#         x = self.lns(x)
#         x = F.dropout(x, p=self.dropout, training=True)
#         x = self.last_layer(x)
            
#         embedding = x
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x


# # GCNII
# class GCNIIModel(nn.Module):

#     def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, alpha, gamma, device):
#         super(GCNIIModel, self).__init__()

#         self.num_layers = num_layers
#         self.dataset = dataset
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.device = device
#         self.gcn2_convs = nn.ModuleList()
#         self.lns = nn.ModuleList()
#         self.mlp_layers = mlp_layers
#         self.alpha = alpha 
#         self.gamma = gamma
#         print("Using GCNII model")
        
#         self.init_layer = nn.Linear(input_dim, hidden_dim)
#         self.last_layer = nn.Linear(hidden_dim, self.dataset.num_classes)

#         for i in range(self.num_layers):
#             self.gcn2_convs.append(GCN2Conv(self.hidden_dim, alpha=0.2, theta=1.5, layer=i+1).to(self.device))
#             self.lns.append(nn.LayerNorm(hidden_dim))
          

#     def forward(self, x, adj_matrix):
#         x = x.to(self.device)
#         num_nodes = x.shape[0]
#         # x_0 = x
#         # x = F.dropout(x, p=self.dropout, training=True)
#         # x = self.init_layer(x)
        
#         adj_matrix = adj_matrix.to(self.device)
    
#         '''
#         adding self-loops and parallel edges in the graph
#         '''

#         adj_matrix = (self.gamma * adj_matrix)
#         adj_matrix = adj_matrix + (torch.eye(num_nodes) * self.alpha).to(self.device)
#         edge_indices = torch.where(adj_matrix != 0)
#         edge_index = torch.stack([edge_indices[0], edge_indices[1]])
#         # edge_weights = adj_matrix[adj_matrix != 0]

#         x = self.init_layer(x)
#         x_0 = x
#         # print("shape ", x.shape)

#         # message propagation through hidden layers
#         for i in range(self.num_layers):
         
#             x = self.gcn2_convs[i](x, x_0, edge_index)

#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = self.lns[i](x)
#                 x = F.dropout(x, p=self.dropout, training=True)
                
#         x = self.last_layer(x)
#         embedding = x
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x
