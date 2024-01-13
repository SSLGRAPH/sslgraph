# 导入包
import pgl
import paddle.nn as nn
from model.contrastive.MVGRL.mvgrl import MVGRL
from paddle.optimizer import Adam
import numpy as np
import paddle
from Trainer.Trainer import Trainer
from dataset.dataset import CoraDataset
from pgl.utils.data import Dataloader


class GCN(nn.Layer):
    """Implement of GCN
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5,
                 norm=None):
        super(GCN, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        self.norm = norm
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation=nn.PReLU(),
                        norm=self.norm))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation=nn.PReLU(),
                        norm=self.norm))
            self.gcns.append(nn.Dropout(self.dropout))
        self.gcns.append(pgl.nn.GCNConv(self.hidden_size, self.num_class))

    def forward(self, graph, feature_numpy):
        graph = graph.tensor()
        feature = paddle.to_tensor(feature_numpy)
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature)
        return feature


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



dataset = CoraDataset()
loader = Dataloader(dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=1
                    )


graph = dataset.graph
encoder1 = GCN(input_size=dataset.features.shape[1], num_class=512, num_layers=1, hidden_size=512, dropout=0.0, norm="true")
encoder2 = GCN(input_size=dataset.features.shape[1], num_class=512, num_layers=1, hidden_size=512, dropout=0.0, norm="none")
encoders = [encoder1, encoder2]
mvgrl = MVGRL(encoders)

optim = Adam(
    learning_rate=1e-3,
    parameters=mvgrl.parameters(),
    weight_decay=0.0)

train = Trainer(full_dataset=loader, dataset=dataset)
train.setup_train_config(p_optim='Adam', p_lr=0.001, p_epoch=30, weight_decay= 1e-5)
train.train_encoder(mvgrl)

