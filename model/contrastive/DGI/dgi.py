from typing import Dict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from augmentation.node_attribute_level import *

from model.contrastive.contrastive import Contrastive


class Discriminator(nn.Layer):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, features, summary):
        score = paddle.matmul(self.linear(features), summary)
        return score


# 将dgi_loss改为继承nn.Layer,从而能够训练其中的参数
class DgiLoss(nn.Layer):
    def __init__(self):
        super(DgiLoss, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()
        self.discriminator = Discriminator(512)
        self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, z1, z2,args: Dict=None):
        summary = F.log_sigmoid(paddle.mean(z1, axis=0))
        pos_score = self.discriminator(z1, summary)
        neg_score = self.discriminator(z2, summary)
        l1 = self.loss_fn(pos_score, paddle.ones_like(pos_score))
        l2 = self.loss_fn(neg_score, paddle.zeros_like(neg_score))
        return l1 + l2


# def dgi_loss(z1, z2):
#     summary = F.log_sigmoid(paddle.sum(z1, axis=0))
#     discriminator = Discriminator(512)
#     pos_score = discriminator(z1, summary)
#     neg_score = discriminator(z2, summary)
#     loss_fn = F.binary_cross_entropy_with_logits
#     l1 = loss_fn(pos_score, paddle.ones_like(pos_score))
#     l2 = loss_fn(neg_score, paddle.zeros_like(neg_score))
#     return l1 + l2


# def corrupt(graph, feature):
#     corrupted_graph = copy.deepcopy(graph)
#     perm = paddle.randperm(feature.shape[0])
#     feature = feature[perm]
#     return corrupted_graph, feature


class DGI(Contrastive):
    def __init__(self, encoders):
        view_fn_1 = lambda graph: graph
        view_fn_2 = NodeAttrRowShuffle()
        views_fn = [view_fn_1, view_fn_2]
        super(DGI, self).__init__(views_fn=views_fn,
                                  encoders=encoders,
                                  loss_layer=DgiLoss(),
                                  mode="node")
