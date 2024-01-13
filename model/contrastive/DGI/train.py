# 导入包
import paddle.nn as nn
from pgl.utils.data import Dataloader

from Trainer.Trainer import Trainer
from dataset.dataset import *
from model.contrastive.DGI.dgi import DGI


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
dgi = DGI(encoders)
print(len(dgi.encoders))
# 配置训练
# optim = Adam(
#     learning_rate=1e-3,
#     parameters=dgi.parameters(),
#     weight_decay=0.0)


train = Trainer(full_dataset=loader, dataset=dataset)
train.setup_train_config(p_optim='Adam', p_lr=0.001, runs=10, p_epoch=300, weight_decay=0.0, batch_szie=256)
train.train_encoder(dgi)
# train.train_classifier(dataset, dgi)
# train.train_classifier(dataset, encoder)
# loss = dgi.train_encoder_one_epoch(data_loader, optim)
# print(loss.item())
