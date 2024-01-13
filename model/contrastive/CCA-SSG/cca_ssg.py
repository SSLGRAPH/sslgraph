import paddle
import paddle.nn as nn

from augmentation import *
from model.contrastive.contrastive import Contrastive


class CCA_SSG_Loss(nn.Layer):
    def __init__(self):
        super(CCA_SSG_Loss, self).__init__()
    def forward(self,z1, z2, args: dict):
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        c = paddle.mm(z1.T, z2)
        c1 = paddle.mm(z1.T, z1)
        c2 = paddle.mm(z2.T, z2)
        n = z1.shape[0]
        c = c / n
        c1 = c1 / n
        c2 = c2 / n

        loss_inv = -paddle.diagonal(c).sum()
        iden = paddle.eye(c.shape[0])
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        loss = loss_inv + args['lamda'] * (loss_dec1 + loss_dec2)
        return loss


class CCA_SSG(Contrastive):
    def __init__(self, encoders, args: dict):
        random_aug = Sequential([NodeAttrMask(0, args['feature_drop_prob']), EdgeRandomAugment(0, args['edge_mask_prob'])])
        view_fn_1 = random_aug
        view_fn_2 = random_aug
        views_fn = [view_fn_1, view_fn_2]
        super(CCA_SSG, self).__init__(views_fn=views_fn,
                                      loss_layer=CCA_SSG_Loss(),
                                      encoders=encoders,
                                      mode="node", args=args)
