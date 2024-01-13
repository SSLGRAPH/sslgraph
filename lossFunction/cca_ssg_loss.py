import paddle
import paddle.nn as nn
import numpy as np

class CustomLoss(nn.Layer):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, z1, z2, lambd):
        N = z1.shape[0]  # 获取节点数量

        c = paddle.mm(z1.T, z2)  # 计算z1和z2之间的相似度矩阵
        c1 = paddle.mm(z1.T, z1)  # 计算z1和自身的相似度矩阵
        c2 = paddle.mm(z2.T, z2)  # 计算z2和自身的相似度矩阵

        c = c / N  # 标准化
        c1 = c1 / N  # 标准化
        c2 = c2 / N  # 标准化

        loss_inv = -paddle.diagonal(c).sum()  # 计算相似度矩阵对角线上的负和
        iden = paddle.to_tensor(np.eye(c.shape[0]))  # 创建一个单位矩阵
        loss_dec1 = (iden - c1).pow(2).sum()  # 计算z1相似度矩阵和单位矩阵之间的平方误差
        loss_dec2 = (iden - c2).pow(2).sum()  # 计算z2相似度矩阵和单位矩阵之间的平方误差

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)  # 总损失是上述三部分的和
        return loss


