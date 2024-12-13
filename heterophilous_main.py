import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import os 

from torch_geometric.loader import GraphSAINTRandomWalkSampler, RandomNodeLoader
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, to_dense_adj, homophily, degree
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor

import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt

from custom_parser import argument_parser
from model_for_new_heterophily import *
from utils import *

from sklearn.metrics import roc_auc_score


# train function
def train(dataset, model, opti_train, train_iter, model_path, device):
    
    best_val_acc = 0.0
    pbar = tqdm(total=train_iter)

    for i in range(train_iter):
       
        model.train()
        opti_train.zero_grad()
        dataset.x = dataset.x.to(device)
        dataset.edge_index = dataset.edge_index.to(device)
        emb, pred = model(dataset.x, dataset.edge_index)
        label = dataset.y.to(device)
        pred_train = pred[dataset.train_mask].to(device)
        label_train = label[dataset.train_mask]
        loss_train = loss_fn(pred_train, label_train) 
        loss_train.backward()
        opti_train.step()

        train_acc, val_acc, test_acc = test(dataset, model)
        # print(f'Iteration: {i:02d} || Train loss: {loss_train.item(): .4f} || Train Acc: {train_acc: .4f} || Valid Acc: {val_acc: .4f} || Test Acc: {test_acc: .4f}')
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        pbar.update(1)
        pbar.set_postfix({"train loss": loss_train.item(), "Train acc": train_acc, "Valid acc": val_acc, "Test acc": test_acc})


@torch.no_grad()
def test(dataset, model):

    model.eval()
    dataset.x = dataset.x.to(device)
    dataset.edge_index = dataset.edge_index.to(device)
    dataset.y = dataset.y.to(device)
    emb, pred = model(dataset.x, dataset.edge_index)
    # print("pred ", pred.shape)

    # print(y_true['train'].shape, "  ", y_true['valid'].shape, "  ", y_true['test'].shape)
    
    if dataname == 'Minesweeper' or dataname == 'Tolokers' or dataname == 'Questions':
        pred = pred[:, 1]
        train_acc = roc_auc_score(y_true = dataset.y[dataset.train_mask].cpu().numpy(), y_score = pred[dataset.train_mask].cpu().numpy()).item()
        valid_acc = roc_auc_score(y_true = dataset.y[dataset.valid_mask].cpu().numpy(), y_score = pred[dataset.valid_mask].cpu().numpy()).item()
        test_acc = roc_auc_score(y_true = dataset.y[dataset.test_mask].cpu().numpy(), y_score = pred[dataset.test_mask].cpu().numpy()).item()
    else:
        pred = pred.argmax(dim=1)
        train_acc = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).float().mean().item()
        valid_acc = (pred[dataset.valid_mask] == dataset.y[dataset.valid_mask]).float().mean().item()
        test_acc = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).float().mean().item()
            

    return train_acc, valid_acc, test_acc


parsed_args = argument_parser().parse_args()

dataname = parsed_args.dataset
train_lr = parsed_args.train_lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
mlp_layers = parsed_args.mlp_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
dropout = parsed_args.dropout
train_weight_decay = parsed_args.train_w_decay
num_splits = parsed_args.num_splits
alpha = parsed_args.alpha
gamma = parsed_args.gamma
device = parsed_args.device

print(parsed_args)

# setting seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# "Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"

dataset = HeterophilousGraphDataset(root='./data', name=dataname)
print(dataset)

print(dataset.x.sum(dim=1))

print("----------------------------------------------")
print("Dataset: ", dataname)
print("Deeper Insights into Heterophilic Graphs")
print("number of hidden layers:", num_layers)
print("-------------------------------------------------------")
print("features ", dataset.x.shape)
print("node labels ", dataset.y.shape)
print("edge index ", dataset.edge_index.shape)
print("mask ", dataset.train_mask.shape, "   ", dataset.val_mask.shape, "   ", dataset.test_mask.shape)
print("----------------------------------------------")

print(f"alpha: {alpha} || gamma: {gamma}")

dataset.num_nodes = dataset.x.shape[0]
dataset.num_edges = dataset.edge_index.shape[1]
# dataset.num_classes = max(dataset.y) + 1


node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
isolated_nodes = torch.sum(torch.eq(node_degrees, 0)).item()
print(f"Isolated nodes: {isolated_nodes} || Total nodes: {dataset.num_nodes}")


# dataset.edge_index = remove_self_loops(dataset.edge_index)[0]
# dataset.edge_index = add_self_loops(dataset.edge_index)[0]
# print("edge index ", dataset.edge_index.shape)

homo_ratio = homophily(dataset.edge_index, dataset.y, method = 'edge')
print("Homophily ratio ", homo_ratio)

# import sys
# sys.exit()

# dataset.edge_index = add_self_loops(dataset.edge_index)[0]
# print(dataset.edge_index.shape)
# degrees = degree(dataset.edge_index[0], num_nodes=dataset.x.shape[0])
# print(sum(degrees))

# row_transform = NormalizeFeatures()
# data = Data(x = dataset.x, edge_index = dataset.edge_index)
# data = row_transform(data)
# dataset.x = data.x
# print(dataset.x.sum(dim=1))
# feat_label_ratio = feature_class_relation(dataset.edge_index, dataset.y, dataset.x)
# print(f"Feature to Label ratio:  {feat_label_ratio.item(): .4f}")

# degree_distribution(dataset.edge_index, dataset.num_nodes, dataname)

# node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
# avg_degrees = node_degrees.sum() / dataset.num_nodes
# print(f"Avg degree: {avg_degrees}")
# import sys 
# sys.exit()

train_mask_set = dataset.train_mask
valid_mask_set = dataset.val_mask
test_mask_set = dataset.test_mask
print(train_mask_set.shape, "  ", valid_mask_set.shape, "  ", test_mask_set.shape)

print("Optimization started....")

test_acc_list = []
for run in range(10):
    print('')
    print(f'Split {run:02d}:')
    
    model_path = 'best_gnn_model.pt'
    model = GCNModel(dataset, num_layers, mlp_layers, dataset.x.shape[1], hidden_dim, dropout, alpha, gamma, device)
    # model = FAModel(dataset, num_layers, mlp_layers, dataset.x.shape[1], hidden_dim, dropout, alpha, gamma, device)
    # model = HPFModel(dataset, num_layers, mlp_layers, dataset.x.shape[1], hidden_dim, dropout, alpha, gamma, device)
    # model = SGCModel(dataset, num_layers, mlp_layers, dataset.x.shape[1], hidden_dim, dropout, alpha, gamma, device)
    model = model.to(device)
    # print(model)

    dataset.train_mask = train_mask_set[:,run]
    # train_idx = torch.where(train_mask != 0)
    dataset.valid_mask = valid_mask_set[:,run]\
    # valid_idx = torch.where(valid_mask != 0)
    dataset.test_mask = test_mask_set[:,run]
    # test_idx = torch.where(test_mask != 0)

    opti_train = torch.optim.Adam(model.parameters(), lr = train_lr, weight_decay = train_weight_decay)

    # dataset = Data(x=dataset.x, edge_index = dataset.edge_index, num_classes=max(dataset.y).item() + 1, num_features = dataset.x.shape[1], y=dataset.y, train_mask=train_idx, valid_mask=valid_idx, test_mask=test_idx)

    # train the model
    train(dataset, model, opti_train, train_iter, model_path, device)


    print('\n**************Evaluation**********************\n')

    model.load_state_dict(torch.load(model_path))
    for i in range(test_iter):
        train_acc, val_acc, test_acc = test(dataset, model)
        print(f'Iteration: {i:02d}, Test Accuracy: {test_acc: .4f}')
        test_acc_list.append(test_acc)
        # visualize(out, y, data_name, num_layers)

    print("-----------------------------------------")

test_acc_list = torch.tensor(test_acc_list)
print(f'Final Test: {test_acc_list.mean():.4f} +- {test_acc_list.std():.4f}')

