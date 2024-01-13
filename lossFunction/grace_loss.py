import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ContrastiveLoss(nn.Layer):
    def __init__(self, fc1, fc2, tau, mean=True, batch_size=0):
        super(ContrastiveLoss, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.tau = tau
        self.mean = mean
        self.batch_size = batch_size

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return paddle.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: paddle.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -paddle.log(
            paddle.diag(between_sim)
            / (refl_sim.sum(1) + between_sim.sum(1) - paddle.diag(refl_sim)))

    def batched_semi_loss(self, z1, z2):
        num_nodes = z1.shape[0]
        num_batches = (num_nodes - 1) // self.batch_size + 1
        f = lambda x: paddle.exp(x / self.tau)
        indices = paddle.arange(0, num_nodes)
        losses = []

        for i in range(num_batches):
            mask = indices[i * self.batch_size:(i + 1) * self.batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(-paddle.log(
                between_sim[:, i * self.batch_size:(i + 1) * self.batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * self.batch_size:(i + 1) * self.batch_size].diag())))

        return paddle.concat(losses)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def forward(self, z1, z2):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if self.batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2)
            l2 = self.batched_semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if self.mean else ret.sum()

        return ret
