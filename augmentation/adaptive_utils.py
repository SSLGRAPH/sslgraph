import networkx as nx
import paddle
import numpy as np
from pgl.utils.transform import to_undirected


def pgl_to_networkx(pgl_graph):
    '''
    将PGL图转换为NetworkX图
    :param pgl_graph:
    :return: networkx图
    '''
    # 创建一个 NetworkX 图对象
    nx_graph = nx.Graph()

    # 获取 PGL 图中的节点数
    num_nodes = pgl_graph.num_nodes

    # 添加节点到 NetworkX 图
    for node_id in range(num_nodes):
        nx_graph.add_node(node_id)

    # 遍历 PGL 图中的边并添加到 NetworkX 图
    for edge in pgl_graph.edges:
        src, dst = edge
        nx_graph.add_edge(src, dst)

    return nx_graph


def pagerank_centrality(graph, damp: float = 0.85, k: int = 10):
    '''Returns the PageRank centrality given the graph
    Args:
        damp (float, optional): Damping factor
        k (int, optional): Number of iterations
        graph:
    '''
    edges = graph.edges
    num_nodes = edges.max().item() + 1
    deg_out = graph.outdegree()
    x = paddle.ones((num_nodes,)).cast('float32')

    for i in range(k):
        edge_msg = x[edges[:, 0]] / deg_out[edges[:, 0]]
        agg_msg = paddle.ones((num_nodes,)).cast('float32')
        agg_msg = paddle.scatter(agg_msg, edges[:, 1], edge_msg, overwrite=False)

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(graph):
    '''
    Returns the eigenvector centrality given the graph
    :param graph:
    :return:
    '''
    nx_graph = pgl_to_networkx(graph)
    x = nx.eigenvector_centrality_numpy(nx_graph)
    x = [x[i] for i in range(graph.num_nodes)]
    return paddle.to_tensor(x, dtype='float32')


def feature_drop_weights(node_feat, node_centrality):
    '''
    用于稀疏的one-hot特征（GCA）
    :param node_centrality: tensor,represnting node centralities
    :param node_feat:tensor
    :return:weights:tensor
    '''
    node_feat = paddle.to_tensor(node_feat)
    node_centrality = paddle.to_tensor(node_centrality, dtype='float32')
    node_feat = node_feat.cast('bool').cast('float32')
    weights = node_feat.t().matmul(node_centrality)
    weights = weights.log()
    weights = (weights.max() - weights) / (weights.max() - weights.mean())
    return weights


def feature_drop_weights_dense(node_feat, node_centrality):
    '''
    用于稠密的特征（GCA）
    :param node_centrality: tensor,represnting node centralities
    :param node_feat:tensor
    :return:weights:tensor
    '''
    node_feat = paddle.to_tensor(node_feat)
    node_centrality = paddle.to_tensor(node_centrality)
    node_feat = node_feat.abs()
    weights = node_feat.matmul(node_centrality)
    weights = weights.log()
    weights = (weights.max() - weights) / (weights.max() - weights.mean())
    return weights


def drop_feature_weighted(feature, weight, p: float, threshold: float = 0.7):
    '''Returns the new node features after probabilistically masking some`
    Args:
        feature (torch.Tensor): Tensor of shape [n_nodes, n_features] representing node features.
        weight (torch.Tensor): Tensor of shape [n_features] representing weights
        p (float): Probability multiplier
        threshold (float): Upper bound probability of masking a feature
    '''
    feature = paddle.to_tensor(feature)
    weight = paddle.to_tensor(weight)
    weight = weight / weight.mean() * p
    weight = paddle.where(weight < threshold, weight, paddle.ones_like(weight) * threshold)
    drop_prob = weight

    drop_mask = paddle.bernoulli(drop_prob).cast('bool')

    feature = feature.clone()
    feature[:, drop_mask] = 0.

    return feature

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float):
    '''Returns the new edge index after probabilistically dropping edges from `edge_index`
    Args:
        edge_index (torch.Tensor): Tensor of shape [2, n_edges]
        edge_weights (torch.Tensor): Tensor of shape [n_edges]
        p (float): Probability multiplier
        threshold (float): Upper bound probability of dropping an edge
    '''
    edge_index = paddle.to_tensor(edge_index)
    edge_weights = paddle.to_tensor(edge_weights)
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = (edge_weights < threshold).where(edge_weights , paddle.ones_like(edge_weights) * threshold)
    sel_mask = paddle.bernoulli(1. - edge_weights).cast('bool')

    return tuple(edge_index[sel_mask])

def degree_drop_weights(graph):
    '''Returns the dropping weight of each edge depending on the degree centrality
    Args:
        edge_index (torch.Tensor): A tensor of shape [2, n_edges]
    :rtype: :class:`Tensor`
    '''
    graph_ = to_undirected(graph.numpy())
    edge_index = graph.edges
    # deg = degree(edge_index_[1])
    deg = graph_.indegree()
    # deg_col = deg[edge_index[1]].to(torch.float32)
    deg_col = deg[edge_index[:, 1]].astype(float)
    s_col = np.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(graph, aggr: str = 'sink', k: int = 10):
    '''Returns the dropping weight of each edge depending on the PageRank centrality
    Args:
        edge_index (torch.Tensor): A tensor of shape [2, n_edges]
    :rtype: :class:`Tensor`
    '''
    edge_index = graph.edges
    pv = pagerank_centrality(edge_index, k=k)
    pv_row = pv[edge_index[0]].cast('float32')
    pv_col = pv[edge_index[1]].cast('float32')
    s_row = paddle.log(pv_row)
    s_col = paddle.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(graph):
    '''Returns the dropping weight of each edge depending on the Eigen vector centrality
    Args:
        edge_index (torch.Tensor): A tensor of shape [2, n_edges]
    :rtype: :class:`Tensor`
    '''
    evc = eigenvector_centrality(graph)
    evc = (evc > 0).where(evc, paddle.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = graph.edges
    s_row, s_col = s[edge_index[:, 0]], s[edge_index[:, 1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())