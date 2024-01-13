import paddle.nn as nn
import paddle as pd
import paddle.nn.functional as F


class bce_loss(nn.Layer):
    def __init__(self):
        super(bce_loss, self).__init__()


    def forward(self, logits,labels):
        print("Logits shape:", logits.shape)
        print("Labels shape:", labels.shape)
        # 假设 out 是 logits，labels 是相应的标签
        loss_function = nn.BCEWithLogitsLoss()
        labels = labels.astype('float32')  # 将 labels 转换为 float32 类型
        loss = loss_function(logits, labels)
        return loss
