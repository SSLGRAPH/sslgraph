import json
import os

import networkx as nx
import numpy as np
import paddle
import pgl




class PPIDataset():
    def __init__(self):
        self.save_path = "/data/data_ppi/"
        self.dataset_str = "ppi"
        self.mode = "train"

    def preprocess_adj(adj):
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d = np.diag(d ** -0.5)
        norm_adj = d.dot(adj).dot(d)
        return norm_adj


    def process(self):
        graph_file = os.path.join(
            self.save_path, "{}_graph.json".format(self.mode)
        )
        label_file = os.path.join(
            self.save_path, "{}_labels.npy".format(self.mode)
        )
        feat_file = os.path.join(
            self.save_path, "{}_feats.npy".format(self.mode)
        )
        graph_id_file = os.path.join(
            self.save_path, "{}_graph_id.npy".format(self.mode)
        )

        g_data = json.load(open(graph_file))
        self.labels = np.load(label_file)
        self.features = np.load(feat_file)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(g_data))
        adj = adj.toarray()
        adj = self.preprocess_adj(adj)
        self.adj = paddle.to_tensor(adj)
        graph_id = np.load(graph_id_file)
        self.graph = pgl.Graph(
            num_nodes=adj.shape[0],
            edges=adj.nonzero().T,
            node_feat={"feature": self.features}
        )

        # lo, hi means the range of graph ids for different portion of the dataset,
        # 20 graphs for training, 2 for validation and 2 for testing.
        lo, hi = 1, 21
        if self.mode == "valid":
            lo, hi = 21, 23
        elif self.mode == "test":
            lo, hi = 23, 25

        graph_masks = []
        self.graphs = []
        for g_id in range(lo, hi):
            g_mask = np.where(graph_id == g_id)[0]
            graph_masks.append(g_mask)
            g = self.graph.subgraph(g_mask)
            self.graphs.append(g)