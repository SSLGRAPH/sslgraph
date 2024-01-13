# 导入包
import json

import pgl
import paddle.nn as nn
from model.contrastive.GRACE.grace import Grace
from paddle.optimizer import Adam
from pgl.utils.data import Dataloader
import numpy as np
from dataset.dataset import *
import paddle
from Trainer.Trainer import Trainer


class GCN(nn.Layer):
    """Implement of GCN
    """
    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5):
        super(GCN, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation=nn.PReLU(),
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation=nn.PReLU(),
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        self.gcns.append(pgl.nn.GCNConv(self.hidden_size, self.num_class))

    def forward(self, graph, feature):
        graph = graph.tensor()
        feature = paddle.to_tensor(feature)
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature)
        return feature


# class Discriminator(nn.Layer):
#     def __init__(self, dim):
#         super(Discriminator, self).__init__()
#         self.fn = nn.Bilinear(dim, dim, 1)
#
#     def forward(self, h1, h2, h3, h4, c1, c2):
#         c_x1 = c1.expand_as(h1).contiguous()
#         c_x2 = c2.expand_as(h2).contiguous()
#
#         # positive
#         sc_1 = self.fn(h2, c_x1).squeeze(1)
#         sc_2 = self.fn(h1, c_x2).squeeze(1)
#
#         # negative
#         sc_3 = self.fn(h4, c_x1).squeeze(1)
#         sc_4 = self.fn(h3, c_x2).squeeze(1)
#
#         logits = paddle.concat((sc_1, sc_2, sc_3, sc_4))
#
#         return logits


def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def load(name, normalized_feature=True):
    if name == 'cora':
        dataset = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(name + " dataset doesn't exists")

    dataset.graph.node_feat["words"] = normalize(dataset.graph.node_feat[
                                                     "words"])

    dataset.graph.tensor()
    train_index = dataset.train_index
    dataset.train_label = paddle.to_tensor(
        np.expand_dims(dataset.y[train_index], -1))
    dataset.train_index = paddle.to_tensor(np.expand_dims(train_index, -1))

    val_index = dataset.val_index
    dataset.val_label = paddle.to_tensor(
        np.expand_dims(dataset.y[val_index], -1))
    dataset.val_index = paddle.to_tensor(np.expand_dims(val_index, -1))

    test_index = dataset.test_index
    dataset.test_label = paddle.to_tensor(
        np.expand_dims(dataset.y[test_index], -1))
    dataset.test_index = paddle.to_tensor(np.expand_dims(test_index, -1))

    return dataset


# 加载数据集
dataset = CoraDataset()
loader = Dataloader(dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=1
                    )
# 定义模型
graph = dataset.graph
encoder = GCN(dataset.features.shape[1], 512, 1, 512, 0.0)
encoders = [encoder]

# # 加载数据集
# dataset = load("cora")
# # 定义模型
# graph = dataset.graph
# encoder = GCN(graph.node_feat["words"].shape[1], 512, 1, 512, 0.0)
#
# encoders = [encoder]
grace = Grace(encoders)

# data_loader = [(graph, graph.node_feat["words"])]
# print('data_loader', data_loader)
train = Trainer(full_dataset=loader, dataset=dataset)
train.setup_train_config(p_optim='Adam', p_lr=0.001, runs=10, p_epoch=300, weight_decay=0.0, batch_szie=256)
train.train_encoder(grace)

