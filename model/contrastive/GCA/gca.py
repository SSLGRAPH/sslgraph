import paddle.nn as nn

from augmentation import *
from lossFunction.gca_loss import loss
from model.contrastive.contrastive import Contrastive


class GCALoss(nn.Layer):
    def __init__(self, dim: int, proj_n_dim: int, tau: float = 0.1):
        super(GCALoss, self).__init__()
        self.fc1 = nn.Linear(dim, proj_n_dim)
        self.fc2 = nn.Linear(proj_n_dim, dim)
        self.tau = tau

    def forward(self, z1, z2, args):
        return loss(z1, z2, self.fc1, self.fc2, self.tau)


class GCA(Contrastive):
    def __init__(self, encoders, args, dim: int, proj_n_dim: int, centrality_measure: str, prob_edge_1: float,
                 prob_edge_2: float,
                 prob_feature_1: float, prob_feature_2: float, tau: float = 0.1, dense: bool = False,
                 p_tau: float = 0.7):
        view_fn_1 = Sequential([AdaEdgeRemove(centrality_measure, delete_ratio=prob_edge_1, threshold=p_tau),
                                AdaNodeAttrMask(centrality_measure, dimension_mask_radio=prob_feature_1,
                                                threshold=p_tau, dense=dense)])
        view_fn_2 = Sequential([AdaEdgeRemove(centrality_measure, delete_ratio=prob_edge_2, threshold=p_tau),
                                AdaNodeAttrMask(centrality_measure, dimension_mask_radio=prob_feature_2,
                                                threshold=p_tau, dense=dense)])
        views_fn = [view_fn_1, view_fn_2]

        super(GCA, self).__init__(views_fn=views_fn,
                                  loss_layer=GCALoss(dim, proj_n_dim, tau),
                                  encoders=encoders,
                                  mode="node")
