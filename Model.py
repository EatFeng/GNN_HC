# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# -*- coding: utf-8 -*-
# @Time     :2024/4/16/16:54
# @Auther   :Eat
# @Function :这里是GNN模型及相关函数
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def train(model, dataset, optimizer, criterion, device):
    model.train()
    loss_accumulate = 0
    for data in dataset:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss_accumulate += loss.item()
        loss.backward()
        optimizer.step()

    return loss_accumulate / len(dataset)


def evaluate(model, dataset, device):
    model.eval()

    correct = 0

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()

    return correct / sum([data.train_mask.sum().item() for data in dataset])


def predict(model, data, device):
    model.eval()
    data = data.to(device)
    out = model(data)
    predicted = out.argmax(dim=1)

    return predicted
