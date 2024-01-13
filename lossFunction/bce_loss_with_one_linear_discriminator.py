import paddle.nn as nn
import paddle.nn.functional as F
import paddle

class Discriminator(nn.Layer):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, features, summary):
        score = paddle.matmul(self.linear(features), summary)
        return score
class BCE_Loss_with_one_discriminator(nn.Layer):
    def __init__(self):
        super(BCE_Loss_with_one_discriminator, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()
        self.discriminator = Discriminator(512)
        self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, z1, z2):
        summary = F.log_sigmoid(paddle.mean(z1, axis=0))
        pos_score = self.discriminator(z1, summary)
        neg_score = self.discriminator(z2, summary)
        l1 = self.loss_fn(pos_score, paddle.ones_like(pos_score))
        l2 = self.loss_fn(neg_score, paddle.zeros_like(neg_score))
        return l1 + l2

