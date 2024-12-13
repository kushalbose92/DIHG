import torch
import torch.nn as nn 
import numpy as np

from statistics import mean

from utils import loss_fn


@torch.no_grad()
def test(model, dataset, adj_matrix, device, is_validation):
    
    if is_validation == True:
        mask = dataset.val_mask
    else:
        mask = dataset.test_mask
        
    correct = 0
    emb, pred = model(dataset.node_features, adj_matrix)
    pred = pred.argmax(dim = 1)
    label = dataset.node_labels.to(device)
    pred = pred[mask].to(device)
    label = label[mask]
    correct = pred.eq(label).sum().item()
    acc = correct / int(mask.sum())

    return acc