import warnings

warnings.filterwarnings("ignore", category=Warning)

import paddle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys
import pgl


class PubmedDataset():
    def __init__(self):
        self.path = "/data/data_pubmed/"
        self.dataset_str = "pubmed"

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    # 归一化特征
    # 按行求均值
    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj(adj):
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d = np.diag(d ** -0.5)
        norm_adj = d.dot(adj).dot(d)
        return norm_adj

    def load_data(self):
        # step 1: 读取 x, y, tx, ty, allx, ally, graph
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(self.path, "ind.{}.{}".format(self.dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        # step 2: 读取测试集索引
        test_idx_reorder = self.parse_index_file(os.path.join(self.path, "ind.{}.test.index".format(self.dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)



        # 获取整个图的所有节点特征
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = self.preprocess_features(features)
        features = features.toarray()

        # 获取整个图的邻接矩阵
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj.toarray()
        adj = self.preprocess_adj(adj)

        # 获取所有节点标签
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.argmax(labels, axis=1)

        # 划分训练集、验证集、测试集索引
        train_idx = list(range(len(y)))
        val_idx = list(range(len(y), len(y) + 500))
        test_idx = test_idx_range.tolist()

        # 转为paddle tensor
        self.adj = paddle.to_tensor(adj)
        self.features = paddle.to_tensor(features)
        self.labels = paddle.to_tensor(labels)
        self.train_idx = paddle.to_tensor(train_idx)
        self.val_idx = paddle.to_tensor(val_idx)
        self.test_idx = paddle.to_tensor(test_idx)
        self.graph = pgl.Graph(
            num_nodes=adj.shape[0],  # 使用邻接矩阵的行数作为节点数
            edges=adj.nonzero().T,  # 从邻接矩阵中提取非零边
            node_feat={"feature": features}
        )



