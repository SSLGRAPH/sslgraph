import copy

import numpy as np
from pgl import Graph
from pgl.utils.transform import add_self_loops, to_undirected

from augmentation.adaptive_utils import feature_drop_weights, feature_drop_weights_dense, drop_feature_weighted, \
    pagerank_centrality, eigenvector_centrality


class NodeAttrRowShuffle():
    def __init__(self):
        pass

    def __call__(self, graph):
        return self.views_fn(graph)

    def do_augment(self, graph):
        new_graph = copy.deepcopy(graph)
        feat_name = list(graph.node_feat.keys())[0]
        node_feat = graph.node_feat[feat_name]
        perm = np.arange(0, graph.num_nodes)
        np.random.shuffle(perm)
        new_graph.node_feat[feat_name] = node_feat[perm]
        new_graph.tensor()
        return new_graph

    def views_fn(self, graph):
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)


class NodeAttrMask():
    def __init__(self, node_mask_radio, dimension_mask_radio):
        self.node_mask_radio = node_mask_radio
        self.dimension_mask_radio = dimension_mask_radio

    def __call__(self, graph):
        return self.views_fn(graph)

    def do_augment(self, graph):
        new_graph = copy.deepcopy(graph)
        feat_name = list(graph.node_feat.keys())[0]
        node_feat = graph.node_feat[feat_name]
        node_mask_index = np.random.choice(graph.num_nodes, int(graph.num_nodes * self.node_mask_radio),
                                           replace=False)
        dimension_mask_index = np.random.choice(node_feat.shape[1],
                                                int(node_feat.shape[1] * self.dimension_mask_radio),
                                                replace=False)
        # 对选中的行进行mask
        if self.node_mask_radio < 1e-9:
            new_graph.node_feat[feat_name][:, dimension_mask_index] = 0
        else:
            new_graph.node_feat[feat_name][node_mask_index, dimension_mask_index] = 0
        return new_graph

    def views_fn(self, graph):
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)


class AdaNodeAttrMask():
    def __init__(self, centrality_measure: str, dimension_mask_radio: float, threshold: float, dense: bool):
        self.centrality_measure = centrality_measure
        self.dimension_mask_radio = dimension_mask_radio
        self.threshold = threshold
        self.dense = dense

    def __call__(self, graph):
        return self.views_fn(graph)

    def __get_node_centrality(self, graph):
        # indegree() or outdegree() returns a tensor of shape [num_nodes]
        if np.count_nonzero(graph.indegree(), axis=0) != graph.num_nodes:
            graph_temp = add_self_loops(graph.numpy())
        else:
            graph_temp = graph
        if self.centrality_measure == 'degree':
            graph_temp = to_undirected(graph_temp)
            node_deg = graph_temp.indegree()
            node_c = node_deg
        elif self.centrality_measure == 'pr':
            node_pr = pagerank_centrality(graph_temp)
            node_c = node_pr
        elif self.centrality_measure == 'evc':
            node_evc = eigenvector_centrality(graph_temp)
            node_c = node_evc
        else:
            # Don't allow masking if centrality measure is not specified
            # GCA official implementation uses a full-one mask, but we mandate the user to remove AdaNodePerturbation from the view_fn
            raise Exception("Centrality measure option '{}' is not available!".format(self.centrality_measure))
        return node_c

    def do_augment(self, graph):
        new_graph = copy.deepcopy(graph)
        feat_name = list(graph.node_feat.keys())[0]
        node_feat = np.copy(graph.node_feat[feat_name])
        node_centrality = self.__get_node_centrality(graph)
        if self.dense:
            feat_weight = feature_drop_weights_dense(node_feat, node_centrality)
        else:
            feat_weight = feature_drop_weights(node_feat, node_centrality)
        node_feat = drop_feature_weighted(node_feat, feat_weight, self.dimension_mask_radio, self.threshold)
        new_graph.node_feat[feat_name] = node_feat
        return new_graph

    def views_fn(self, graph):
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)
