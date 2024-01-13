import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class InfoGraphLoss(nn.Layer):
    def __init__(self, measure):
        super(InfoGraphLoss, self).__init__()
        self.measure = measure
        self.log_2 = np.log(2.)

    def get_positive_expectation(self, p_samples, average=True):
        """Computes the positive part of a divergence / difference."""
        if self.measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif self.measure == 'JSD':
            Ep = self.log_2 - F.softplus(- p_samples)
        elif self.measure == 'X2':
            Ep = p_samples ** 2
        elif self.measure == 'KL':
            Ep = p_samples + 1.
        elif self.measure == 'RKL':
            Ep = -paddle.exp(-p_samples)
        elif self.measure == 'DV':
            Ep = p_samples
        elif self.measure == 'H2':
            Ep = 1. - paddle.exp(-p_samples)
        elif self.measure == 'W1':
            Ep = p_samples

        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, average=True):
        """Computes the negative part of a divergence / difference."""
        if self.measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif self.measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - self.log_2
        elif self.measure == 'X2':
            Eq = -0.5 * ((paddle.sqrt(q_samples ** 2) + 1.) ** 2)
        elif self.measure == 'KL':
            Eq = paddle.exp(q_samples)
        elif self.measure == 'RKL':
            Eq = q_samples - 1.
        elif self.measure == 'H2':
            Eq = paddle.exp(q_samples) - 1.
        elif self.measure == 'W1':
            Eq = q_samples

        if average:
            return Eq.mean()
        else:
            return Eq

    def forward(self, l_enc, g_enc, batch, mask=None):
        num_graphs = g_enc.shape[0]
        num_nodes = l_enc.shape[0]
        max_nodes = num_nodes // num_graphs

        pos_mask = paddle.zeros((num_nodes, num_graphs))
        neg_mask = paddle.ones((num_nodes, num_graphs))
        msk = paddle.ones((num_nodes, num_graphs))
        for nodeidx, graphidx in enumerate(batch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.
        if mask is not None:
            for idx, m in enumerate(mask):
                msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.
            res = paddle.mm(l_enc, g_enc.t()) * msk
        else:
            res = paddle.mm(l_enc, g_enc.t())
        E_pos = self.get_positive_expectation(res * pos_mask, average=False).sum()
        E_pos = E_pos / num_nodes
        E_neg = self.get_negative_expectation(res * neg_mask, average=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))
        return E_neg - E_pos
