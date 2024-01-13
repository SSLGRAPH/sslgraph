from augmentation import Sequential, AdaEdgeRemove
from model.contrastive.contrastive import Contrastive
from augmentation.node_attribute_level import NodeAttrMask, AdaNodeAttrMask, NodeAttrRowShuffle
import paddle.nn as nn
import paddle
import paddle.nn.functional as F
import copy
import numpy as np
from typing import Dict


class MLP(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        return self.fc2(z)


class GraceLoss(nn.Layer):
    def __init__(self):
        super(GraceLoss, self).__init__()
        self.proj = MLP(512, 512)
        self.temp = 0.5

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        s = paddle.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2, temp):
        # calculate SimCLR loss
        f = lambda x: paddle.exp(x / self.temp)
        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs
        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - paddle.diag(refl_sim)
        loss = -paddle.log(paddle.diag(between_sim) / x1)
        return loss

    def forward(self, h1, h2,args: Dict=None):
        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        # get loss
        l1 = self.get_loss(z1, z2, temp=1)
        l2 = self.get_loss(z2, z1, temp=1)
        ret = (l1 + l2) * 0.5
        return ret.mean()


def corrupt(graph, feature):
    corrupted_graph = copy.deepcopy(graph)
    perm = paddle.randperm(feature.shape[0])
    feature = feature[perm]
    return corrupted_graph, feature


def aug(graph, x):
    feat = drop_feature(x, 0.5)
    ng = copy.deepcopy(graph)
    return ng, feat


def drop_feature(x, drop_prob):
    drop_mask = (
            paddle.empty((x.shape[1],), dtype='float32').uniform_(0, 1)
            < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = paddle.FloatTensor(np.ones(E) * mask_prob)
    masks = paddle.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


class Grace(Contrastive):
    def __init__(self, encoders):
        view_fn_1 = AdaNodeAttrMask(centrality_measure='degree', dimension_mask_radio=0.4, threshold=0.7, dense=False)
        view_fn_2 = AdaEdgeRemove(centrality_measure='degree', delete_ratio=0.4, threshold=0.7)
        # view_fn_2 = AdaEdgeRemove(centrality_measure='degree', delete_ratio=0.4, threshold=0.7)
        views_fn = [view_fn_1, view_fn_2]
        super(Grace, self).__init__(views_fn=views_fn, encoders=encoders, loss_layer=GraceLoss(), mode="node")
