import warnings

warnings.filterwarnings("ignore", category=Warning)

import paddle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys




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


def load_data_cora(path, dataset_str = "cora"):
    # step 1: 读取 x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引
    test_idx_reorder = parse_index_file(os.path.join(path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)


    # 获取整个图的所有节点特征
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.toarray()

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.toarray()
    adj = preprocess_adj(adj)

    # 获取所有节点标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    # 划分训练集、验证集、测试集索引
    train_idx = list(range(len(y)))
    val_idx = list(range(len(y), len(y) + 500))
    test_idx = test_idx_range.tolist()

    # 转为paddle tensor
    adj = paddle.to_tensor(adj)
    features = paddle.to_tensor(features)
    labels = paddle.to_tensor(labels)
    train_idx = paddle.to_tensor(train_idx)
    val_idx = paddle.to_tensor(val_idx)
    test_idx = paddle.to_tensor(test_idx)
    graph = pgl.Graph(
        num_nodes=adj.shape[0],  # 使用邻接矩阵的行数作为节点数
        edges=adj.nonzero().T,  # 从邻接矩阵中提取非零边
        node_feat={"feature": features}
    )

    return adj, features, labels, train_idx, val_idx, test_idx, graph

def load_data_pubmed(path, dataset_str = "pubmed"):
    # step 1: 读取 x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引
    test_idx_reorder = parse_index_file(os.path.join(path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)


    # 获取整个图的所有节点特征
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.toarray()

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.toarray()
    adj = preprocess_adj(adj)

    # 获取所有节点标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    # 划分训练集、验证集、测试集索引
    train_idx = list(range(len(y)))
    val_idx = list(range(len(y), len(y) + 500))
    test_idx = test_idx_range.tolist()

    # 转为paddle tensor
    adj = paddle.to_tensor(adj)
    features = paddle.to_tensor(features)
    labels = paddle.to_tensor(labels)
    train_idx = paddle.to_tensor(train_idx)
    val_idx = paddle.to_tensor(val_idx)
    test_idx = paddle.to_tensor(test_idx)

    return adj, features, labels, train_idx, val_idx, test_idx


def load_data_citeseer(path, dataset_str = "citeseer"):
    # step 1: 读取 x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引
    test_idx_reorder = parse_index_file(os.path.join(path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    # Find isolated nodes, add them as zero-vecs into the right position
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
    features = preprocess_features(features)
    features = features.toarray()

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.toarray()
    adj = preprocess_adj(adj)

    # 获取所有节点标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    # 划分训练集、验证集、测试集索引
    train_idx = list(range(len(y)))
    val_idx = list(range(len(y), len(y) + 500))
    test_idx = test_idx_range.tolist()

    # 转为paddle tensor
    adj = paddle.to_tensor(adj)
    features = paddle.to_tensor(features)
    labels = paddle.to_tensor(labels)
    train_idx = paddle.to_tensor(train_idx)
    val_idx = paddle.to_tensor(val_idx)
    test_idx = paddle.to_tensor(test_idx)

    return adj, features, labels, train_idx, val_idx, test_idx

def load_data_ppi(mode, save_path):
    # 构建文件路径
    graph_file = os.path.join(save_path, "{}_graph.json".format(mode))
    label_file = os.path.join(save_path, "{}_labels.npy".format(mode))
    feat_file = os.path.join(save_path, "{}_feats.npy".format(mode))
    graph_id_file = os.path.join(save_path, "{}_graph_id.npy".format(mode))

    g_data = json.load(open(graph_file))
    _labels = np.load(label_file)
    _feats = np.load(feat_file)
    graph = nx.DiGraph(json_graph.node_link_graph(g_data))
    graph_id = np.load(graph_id_file)

    # 设置训练、验证和测试的范围
    lo, hi = 1, 21
    if mode == "valid":
        lo, hi = 21, 23
    elif mode == "test":
        lo, hi = 23, 25

    graph_masks = []
    graphs = []
    feats_list = []  # 存储节点特征的列表
    labels_list = []  # 存储节点标签的列表
    for g_id in range(lo, hi):
        g_mask = np.where(graph_id == g_id)[0]
        graph_masks.append(g_mask)
        g = graph.subgraph(g_mask)

        # 创建特征和标签列表
        feats = np.array(_feats[g_mask], dtype=np.float32)
        labels = np.array(_labels[g_mask], dtype=np.float32)

        feats_list.append(feats)
        labels_list.append(labels)

        graphs.append(g)

    return graphs, feats_list, labels_list




















