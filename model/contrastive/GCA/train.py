import paddle.nn as nn
import paddle.nn.functional as F
from pgl.utils.data import Dataloader

from Trainer.Trainer import Trainer
from dataset.dataset import *
from model.contrastive.GCA.gca import GCA


class Encoder(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=pgl.nn.GCNConv, k: int = 2,
                 skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.LayerList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.LayerList(self.conv)

            self.activation = activation

    def forward(self, graph: pgl.Graph, x):
        graph = graph.tensor()
        x = paddle.to_tensor(x)
        # x = graph.node_feat[list(graph.node_feat.keys())[0]]
        edge_index = graph.edges
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](graph, x))
            return x
        else:
            h = self.activation(self.conv[0](graph, x))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = paddle.sum(paddle.stack(hs), axis=0)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': paddle.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


dataset1 = CoraDataset()
dataset2 = CiteseerDataset()
loader = Dataloader(dataset1,
                    batch_size=256,
                    shuffle=False,
                    num_workers=1
                    )

param = {
    "learning_rate": 0.0005,
    "num_hidden": 256,
    "num_proj_hidden": 256,
    "activation": "rrelu",
    "drop_edge_rate_1": 0.1,
    "drop_edge_rate_2": 0.2,
    "drop_feature_rate_1": 0.2,
    "drop_feature_rate_2": 0.1,
    "tau": 0.4,
    "num_epochs": 2,
    'num_layers': 2,
    'weight_decay': 1e-5,
    'drop_scheme': 'degree',
    'base_model': 'GCNConv',
}

encoder = Encoder(dataset1.features.shape[1], param['num_hidden'], get_activation(param['activation']),
                  k=param['num_layers'])
encoders = [encoder, encoder]

model = GCA(encoders=encoders,
            dim=param['num_hidden'],
            proj_n_dim=param['num_proj_hidden'], centrality_measure=param['drop_scheme'],
            prob_edge_1=param['drop_edge_rate_1'],
            prob_edge_2=param['drop_edge_rate_2'],
            prob_feature_1=param['drop_feature_rate_1'],
            prob_feature_2=param['drop_feature_rate_2'],
            tau=param['tau'],
            args={})

train = Trainer(full_dataset=loader, dataset=dataset1)
train.setup_train_config(p_optim='Adam',
                         p_lr=param['learning_rate'],
                         weight_decay=param['weight_decay'],
                         p_epoch=param['num_epochs'])
train.train_encoder(model)
