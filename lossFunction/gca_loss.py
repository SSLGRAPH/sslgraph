import paddle
import paddle.nn.functional as F
from typing import Optional


def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return paddle.mm(z1, z2.t())


def semi_loss(z1, z2, tau):
    f = lambda x: paddle.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -paddle.log(
        paddle.diag(between_sim)
        / (refl_sim.sum(1) + between_sim.sum(1) - paddle.diag(refl_sim)))


def batched_semi_loss(z1, z2, tau, batch_size):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: paddle.exp(x / tau)
    indices = paddle.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))
        between_sim = f(sim(z1[mask], z2))

        losses.append(-paddle.log(
            between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (refl_sim.sum(1) + between_sim.sum(1)
               - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return paddle.cat(losses)


def projection(z, fc1, fc2):
    z = F.elu(fc1(z))
    return fc2(z)


def loss(z1, z2, fc1, fc2, tau, mean=True, batch_size: Optional[int] = 0):
    h1 = projection(z1, fc1, fc2)
    h2 = projection(z2, fc1, fc2)

    if batch_size == 0:
        l1 = semi_loss(h1, h2, tau)
        l2 = semi_loss(h2, h1, tau)
    else:
        l1 = batched_semi_loss(h1, h2, tau, batch_size)
        l2 = batched_semi_loss(h2, h1, tau, batch_size)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret
