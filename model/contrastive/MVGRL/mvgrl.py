import paddle as pd
import paddle.nn
import paddle.nn as nn
import pgl
import numpy as np

from model.contrastive.contrastive import Contrastive
from lossFunction.bce_loss import bce_loss
from augmentation.topology_level import *
from lossFunction.bce_loss_with_one_linear_discriminator import BCE_Loss_with_one_discriminator
from augmentation.topology_level import GraphDiffusion


class Discriminator(nn.Layer):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1)
        c_x2 = c2.expand_as(h2)

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = pd.concat((sc_1, sc_2, sc_3, sc_4))

        return logits


class MVGRL(Contrastive):
    def __init__(self, encoders):

        view_fn_1 = GraphDiffusion()
        view_fn_2 = GraphDiffusion()
        views_fn = [view_fn_1, view_fn_2]
        self.encoders=encoders

        super(MVGRL, self).__init__(views_fn=views_fn,
                                  encoders=encoders,
                                  loss_layer=bce_loss(),
                                  mode="node_graph")
        self.pooling = pgl.nn.pool.GraphPool(pool_type='MEAN')
        self.disc = Discriminator(512)
        self.act_fn = nn.Sigmoid()

        self.encoder1 = self.encoders[0]
        self.encoder2 = self.encoders[1]

    def forward(self, graph, diff_graph, feat, shuf_feat, edge_weight):

        h1 = self.encoder1(graph, feat)
        #h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)
        h2 = self.encoder2(diff_graph, feat)

        h3 = self.encoder1(graph, shuf_feat)
        #h4 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)
        h4 = self.encoder2(diff_graph, shuf_feat)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))
        # c2 = self.act_fn(self.pooling(diff_graph, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)
        return out


