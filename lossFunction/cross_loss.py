import paddle
def loss(logits,labels):
    loss=paddle.nn.CrossEntropyLoss()
    return loss(logits,labels)
