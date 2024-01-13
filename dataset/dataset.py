import warnings
import json

warnings.filterwarnings("ignore", category=Warning)

import paddle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys
import pgl
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.logger import log


class CoraDataset(BaseDataset):
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/data_cora')
        self.dataset_str = 'cora'
        self.load_data()

    def parse_index_file(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    # 归一化特征
    # 按行求均值
    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj(self, adj):
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d = np.diag(d ** -0.5)
        norm_adj = d.dot(adj).dot(d)
        return norm_adj

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        :param idx:
        :return: graph and labels of the graph nodes
        """
        if idx > 0:
            raise IndexError
        return self.graph, self.labels

    def load_data(self):
        # step 1: 读取 x, y, tx, ty, allx, ally, graph
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(
                    os.path.join(self.path, "ind.{}.{}".format(self.dataset_str, names[i])),
                    'rb') as f:
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
        self.train_index = paddle.to_tensor(train_idx)
        # dimensions of train_label must be 2
        self.train_label = paddle.to_tensor(labels[train_idx].reshape(-1, 1))
        self.val_index = paddle.to_tensor(val_idx)
        self.val_label = paddle.to_tensor(labels[val_idx].reshape(-1, 1))
        self.test_index = paddle.to_tensor(test_idx)
        self.test_label = paddle.to_tensor(labels[test_idx].reshape(-1, 1))
        self.num_classes = labels.max() + 1
        self.graph = pgl.Graph(
            num_nodes=adj.shape[0],  # 使用邻接矩阵的行数作为节点数
            edges=np.vstack(adj.nonzero()).T,  # 从邻接矩阵中提取非零边
            node_feat={"feature": features}
        )
        # self.graph = self.graph.numpy()
        # print(self.graph)


class PubmedDataset(BaseDataset):
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data_pubmed")
        self.dataset_str = "pubmed"
        self.load_data()

    def parse_index_file(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    # 归一化特征
    # 按行求均值
    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj(self, adj):
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d = np.diag(d ** -0.5)
        norm_adj = d.dot(adj).dot(d)
        return norm_adj

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        :param idx:
        :return: graph and labels of the graph nodes
        """
        if idx > 0:
            raise IndexError
        return self.graph, self.labels

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
            edges=np.vstack(adj.nonzero()).T,  # 从邻接矩阵中提取非零边
            node_feat={"feature": features}
        )


class CiteseerDataset(BaseDataset):
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data_citeseer")
        self.dataset_str = "citeseer"
        self.load_data()

    def parse_index_file(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    # 归一化特征
    # 按行求均值
    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj(self, adj):
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d = np.diag(d ** -0.5)
        norm_adj = d.dot(adj).dot(d)
        return norm_adj

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        :param idx:
        :return: graph and labels of the graph nodes
        """
        if idx > 0:
            raise IndexError
        return self.graph, self.labels

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

        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

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
        self.adj = paddle.to_tensor(adj, dtype="float32")
        self.features = paddle.to_tensor(features, dtype="float32")
        self.labels = paddle.to_tensor(labels, dtype="float32")
        self.train_idx = paddle.to_tensor(train_idx, dtype="float32")
        self.val_idx = paddle.to_tensor(val_idx, dtype="float32")
        self.test_idx = paddle.to_tensor(test_idx, dtype="float32")
        self.graph = pgl.Graph(
            num_nodes=adj.shape[0],  # 使用邻接矩阵的行数作为节点数
            edges=np.vstack(adj.nonzero()).T,  # 从邻接矩阵中提取非零边
            node_feat={"feature": features}
        )
        print(self.graph)


class PPIDataset(BaseDataset):
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data_ppi")
        self.dataset_name = "ppi"
        self.mode = "train"
        self.process()

    def __len__(self):
        """return the number of graphs"""
        return len(self.graphs)

    def __getitem__(self, idx):
        """getitem"""
        return self.graphs[idx]

    def preprocess_adj(self, adj):
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d = np.diag(d ** -0.5)
        norm_adj = d.dot(adj).dot(d)
        return norm_adj

    def process(self):
        graph_file = os.path.join(
            self.data_path, "{}_graph.json".format(self.mode)
        )
        label_file = os.path.join(
            self.data_path, "{}_labels.npy".format(self.mode)
        )
        feat_file = os.path.join(
            self.data_path, "{}_feats.npy".format(self.mode)
        )
        graph_id_file = os.path.join(
            self.data_path, "{}_graph_id.npy".format(self.mode)
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
            edges=np.vstack(adj.nonzero()).T,
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


class GINDataset(BaseDataset):
    """Dataset for Graph Isomorphism Network (GIN)
    Adapted from https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip.
    """

    def __init__(self,
                 data_path,
                 dataset_name,
                 self_loop,
                 degree_as_nlabel=False):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.self_loop = self_loop
        self.degree_as_nlabel = degree_as_nlabel

        self.graph_list = []
        self.glabel_list = []

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.num_graph = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False

        self._load_data()

    def __len__(self):
        """return the number of graphs"""
        return len(self.graph_list)

    def __getitem__(self, idx):
        """getitem"""
        return self.graph_list[idx], self.glabel_list[idx]

    def _load_data(self):
        """Loads dataset
        """
        filename = os.path.join(self.data_path, self.dataset_name,
                                "%s.txt" % self.dataset_name)
        log.info("loading data from %s" % filename)

        with open(filename, 'r') as reader:
            # first line --> N, means total number of graphs
            self.num_graph = int(reader.readline().strip())

            for i in range(self.num_graph):
                if (i + 1) % int(self.num_graph / 10) == 0:
                    log.info("processing graph %s" % (i + 1))
                graph = dict()
                # second line --> [num_node, label]
                # means [node number of a graph, class label of a graph]
                grow = reader.readline().strip().split()
                n_nodes, glabel = [int(w) for w in grow]

                # relabel graphs
                if glabel not in self.glabel_dict:
                    mapped = len(self.glabel_dict)
                    self.glabel_dict[glabel] = mapped

                graph['num_nodes'] = n_nodes
                self.glabel_list.append(self.glabel_dict[glabel])

                nlabels = []
                node_features = []
                num_edges = 0
                edges = []

                for j in range(graph['num_nodes']):
                    slots = reader.readline().strip().split()

                    # handle edges and node feature(if has)
                    tmp = int(slots[
                                  1]) + 2  # tmp == 2 + num_edges of current node
                    if tmp == len(slots):
                        # no node feature
                        nrow = [int(w) for w in slots]
                        nfeat = None
                    elif tmp < len(slots):
                        nrow = [int(w) for w in slots[:tmp]]
                        nfeat = [float(w) for w in slots[tmp:]]
                        node_features.append(nfeat)
                    else:
                        raise Exception('edge number is not correct!')

                    # relabel nodes if is has labels
                    # if it doesn't have node labels, then every nrow[0] == 0
                    if not nrow[0] in self.nlabel_dict:
                        mapped = len(self.nlabel_dict)
                        self.nlabel_dict[nrow[0]] = mapped

                    nlabels.append(self.nlabel_dict[nrow[0]])
                    num_edges += nrow[1]
                    edges.extend([(j, u) for u in nrow[2:]])

                    if self.self_loop:
                        num_edges += 1
                        edges.append((j, j))

                if node_features != []:
                    node_features = np.stack(node_features)
                    graph['attr'] = node_features
                    self.nattrs_flag = True
                else:
                    node_features = None
                    graph['attr'] = node_features

                graph['nlabel'] = np.array(
                    nlabels, dtype="int64").reshape(-1, 1)
                if len(self.nlabel_dict) > 1:
                    self.nlabels_flag = True

                graph['edges'] = edges
                assert num_edges == len(edges)

                g = pgl.Graph(
                    num_nodes=graph['num_nodes'],
                    edges=graph['edges'],
                    node_feat={
                        'nlabel': graph['nlabel'],
                        'attr': graph['attr']
                    })

                self.graph_list.append(g)

                # update statistics of graphs
                self.n += graph['num_nodes']
                self.m += num_edges

        # if no attr
        if not self.nattrs_flag:
            log.info('there are no node features in this dataset!')
            label2idx = {}
            # generate node attr by node degree
            if self.degree_as_nlabel:
                log.info('generate node features by node degree...')
                nlabel_set = set([])
                for g in self.graph_list:
                    g.node_feat['nlabel'] = g.indegree()
                    # extracting unique node labels
                    nlabel_set = nlabel_set.union(set(g.node_feat['nlabel']))
                    g.node_feat['nlabel'] = g.node_feat['nlabel'].reshape(-1,
                                                                          1)

                nlabel_set = list(nlabel_set)
                # in case the labels/degrees are not continuous number
                self.ndegree_dict = {
                    nlabel_set[i]: i
                    for i in range(len(nlabel_set))
                }
                label2idx = self.ndegree_dict
            # generate node attr by node label
            else:
                log.info('generate node features by node label...')
                label2idx = self.nlabel_dict

            for g in self.graph_list:
                attr = np.zeros((g.num_nodes, len(label2idx)))
                idx = [
                    label2idx[tag]
                    for tag in g.node_feat['nlabel'].reshape(-1, )
                ]
                attr[:, idx] = 1
                g.node_feat['attr'] = attr.astype("float32")

        # after load, get the #classes and #dim
        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graph_list[0].node_feat['attr'][0])

        message = "finished loading data\n"
        message += """
                    num_graph: %d
                    num_graph_class: %d
                    total_num_nodes: %d
                    node Classes: %d
                    node_features_dim: %d
                    num_edges: %d
                    edge_classes: %d
                    Avg. of #Nodes: %.2f
                    Avg. of #Edges: %.2f
                    Graph Relabeled: %s
                    Node Relabeled: %s
                    Degree Relabeled(If degree_as_nlabel=True): %s""" % (
            self.num_graph,
            self.gclasses,
            self.n,
            self.nclasses,
            self.dim_nfeats,
            self.m,
            self.eclasses,
            self.n / self.num_graph,
            self.m / self.num_graph,
            self.glabel_dict,
            self.nlabel_dict,
            self.ndegree_dict,)
        log.info(message)
