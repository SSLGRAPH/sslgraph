import paddle
import pgl
from paddle import nn
from dataset.dataset import *
from pgl.utils.data import Dataloader
from cca_ssg import CCA_SSG
from Trainer.Trainer import Trainer


class GCN(nn.Layer):
    """Implement of GCN
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.0):
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


class Encoder(nn.Layer):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.backbone = GCN(input_size=in_dim, hidden_size=hid_dim, num_class=out_dim, num_layers=n_layers)

    def forward(self, graph1, feat1):
        h1 = self.backbone(graph1, feat1)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        return z1


cora_dataset = CoraDataset()
loader = Dataloader(
    cora_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=1
)

paddle.seed(19)

params = {
    "learning_rate": 0.0005,
    "feature_drop_prob": 0.15,
    "edge_mask_prob": 0.6,
    "num_epochs": 150,
    'num_layers': 3,
    'weight_decay': 0.0,
}

encoder = GCN(input_size=cora_dataset.features.shape[1], num_class=512, num_layers=2, hidden_size=512, dropout=0.0)

encoders = [encoder]

args = {
    'feature_drop_prob': params['feature_drop_prob'],
    'edge_mask_prob': params['edge_mask_prob'],
    'lamda': 1e-3,
}

model = CCA_SSG(encoders=encoders, args=args)

trainer = Trainer(full_dataset=loader, dataset=cora_dataset)

trainer.setup_train_config(p_optim='Adam',
                           p_lr=params['learning_rate'],
                           weight_decay=params['weight_decay'],
                           p_epoch=params['num_epochs'])
trainer.train_encoder(model)
