import numpy as np
import tqdm
import pgl
import paddle
import paddle.nn as nn
from pgl.utils.logger import log

from typing import List, Callable, Dict

from tqdm import trange


class Contrastive(nn.Layer):
    def __init__(self, views_fn: List[Callable], encoders: List[nn.Layer], loss_layer: nn.Layer, mode: str,
                 device=None, args: Dict=None):
        super(Contrastive, self).__init__()
        self.views_fn = views_fn
        self.encoders = encoders
        self.loss_layer = loss_layer
        self.mode = mode
        self.args = args
        self.mode_map: Dict[str, Callable] = {
            "node": self.train_node_encoder_one_epoch,
            "graph": self.train_graph_encoder_one_epoch,
            "node_graph": self.train_node_graph_encoder_one_epoch
        }
        # 需要训练的参数，在initialize中初始化
        self.params = []
        self.initialize()
        self.train_fn = self.mode_map[mode]

        if device is None:
            self.device = paddle.set_device('gpu' if paddle.device.is_compiled_with_cuda() else 'cpu')
        elif isinstance(device, int):
            self.device = paddle.set_device('gpu:%d' % device)
        else:
            self.device = device

    def initialize(self):
        # 合法性检查
        assert len(self.encoders) == 1 or len(self.views_fn) == len(self.encoders), \
            "Encoders and views mismatch!"
        assert self.mode in self.mode_map, \
            f"Invalid mode: {self.mode}. Available modes: {', '.join(self.mode_map.keys())}"

        # initialize,确定需要训练的参数
        if len(self.encoders) == 1:
            # 只有一个encoder,训练参数只添加一次
            encoder = self.encoders[0]
            self.params.extend(encoder.parameters())
            encoder.train()
            self.encoders = [encoder] * len(self.views_fn)
        else:
            [encoder.train() for encoder in self.encoders]
            [self.params.extend(encoder.parameters()) for encoder in self.encoders]

        self.loss_layer.train()
        self.params.extend(self.loss_layer.parameters())

    def train_encoder_one_epoch(self, data_loader, optimizer):
        return self.train_fn(data_loader, optimizer)

    def train_node_encoder_one_epoch(self, data_loader, optimizer):
        epoch_loss = 0.0
        for data in data_loader:
            optimizer.clear_grad()
            views = [v_fn(data[0][0]) for v_fn in self.views_fn]
            zs_n = []
            for view, enc in zip(views, self.encoders):
                # print(view)
                z_n = enc(view, view.node_feat[list(view.node_feat.keys())[0]])
                zs_n.append(z_n)
            loss = self.loss_layer(zs_n[0], zs_n[1] , args=self.args)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        return epoch_loss

    def train_graph_encoder_one_epoch(self, data_loader, optimizer):
        pass

    def train_node_graph_encoder_one_epoch(self, data_loader, optimizer):
        epoch_loss = 0.0
        for data in data_loader:
            print("data",data)

            feat = data[0][0].node_feat["feature"]
            print("feat shape:", feat.shape)
            v_fn1 = self.views_fn[0]
            v_fn2 = self.views_fn[1]
            optimizer.clear_grad()
            graph = v_fn1(data[0][0])
            diff_graph = v_fn2(data[0][0])

            n_node = graph.num_nodes

            lbl1 = paddle.ones(n_node * 2)
            lbl2 = paddle.zeros(n_node * 2)
            lbl = paddle.concat((lbl1, lbl2))

            shuf_idx = np.random.permutation(n_node.item())
            shuf_idx_tensor = paddle.to_tensor(shuf_idx, dtype='int32')
            shuf_feat = feat[shuf_idx_tensor].astype('float32')


            num_edges = graph.num_edges  # 获取边的数量
            edge_weight = paddle.full(shape=[num_edges], fill_value=1.0, dtype='float32')

            feat = feat.astype('float32')
            out = self.forward(graph, diff_graph, feat, shuf_feat, edge_weight)

            loss = self.loss_layer(out, lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        return epoch_loss

    # 模型需要训练的所有参数
    def parameters(self, include_sublayers=True):
        return self.params
