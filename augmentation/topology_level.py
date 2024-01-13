import copy

import numpy as np
import pgl
import paddle
from pgl import Graph
from pgl.utils.transform import add_self_loops

from .adaptive_utils import degree_drop_weights, evc_drop_weights, pr_drop_weights, drop_edge_weighted


class EdgeRandomAugment():
    def __init__(self, add_ratio=0.1, delete_ratio=0.1):
        """
        Initialize the EdgeRandomAugment class.

        Parameters:
        - add_ratio (float): Proportion of edges to be added.
        - delete_ratio (float): Proportion of edges to be removed.
        """
        self.add_ratio = add_ratio
        self.delete_ratio = delete_ratio

    def __call__(self, graph):
        """
        Make the class callable. This returns the augmented graph.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        return self.views_fn(graph)

    def do_augment(self, graph):
        """
        Perform the augmentation on the graph.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        edges = graph.edges
        num_edges = len(edges)

        # Delete a fraction of edges based on delete_ratio
        num_delete = int(num_edges * self.delete_ratio)
        delete_indices = np.random.choice(num_edges, num_delete, replace=False)
        remaining_edges = np.delete(edges, delete_indices, axis=0)

        # Add a fraction of edges based on add_ratio
        num_add = int(num_edges * self.add_ratio)
        new_edges = []
        for _ in range(num_add):
            src = np.random.randint(0, graph.num_nodes)
            dst = np.random.randint(0, graph.num_nodes)
            while (src, dst) in remaining_edges or src == dst:
                src = np.random.randint(0, graph.num_nodes)
                dst = np.random.randint(0, graph.num_nodes)
            new_edges.append((src, dst))

        if self.add_ratio < 1e-9:
            final_edges = remaining_edges
        else:
            final_edges = np.vstack([remaining_edges, new_edges])


        # Create a new graph with the modified edges
        new_graph = pgl.Graph(num_nodes=graph.num_nodes, edges=final_edges, node_feat=graph.node_feat,
                              edge_feat=graph.edge_feat)

        return new_graph

    def views_fn(self, graph):
        """
        Wrapper function to call the augmentation function.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)


# class EdgeRandomRemove(EdgeRandomAugment):
#     def __init__(self, delete_ratio=0.1):
#         """
#         Initialize the EdgeRandomRemove class.
#
#         Parameters:
#         - delete_ratio (float): Proportion of edges to be removed.
#         """
#         # Set add_ratio to 0 and only use delete_ratio
#         super(EdgeRandomRemove, self).__init__(add_ratio=0, delete_ratio=delete_ratio)

class AdaEdgeRemove():
    def __init__(self, centrality_measure: str, delete_ratio: float, threshold: float):
        """
        Initialize the EdgeRandomRemove class.

        Parameters:
        - centrality_measure (str): The method to compute node importance. Options: 'degree', 'pr', 'evc'.
        - delete_ratio (float): Maximum proportion of edges to be removed.
        """
        self.centrality_measure = centrality_measure
        self.delete_ratio = delete_ratio
        self.threshold = threshold

    def __call__(self, graph):
        """
        Make the class callable. This returns the graph with edges removed.

        Parameters:
        - graph (pgl.Graph): Input graph from which edges will be removed.

        Returns:
        - pgl.Graph: Graph with edges removed.
        """
        return self.views_fn(graph)

    def __get_edge_weights(self, graph):
        """
        Compute node importance based on the specified centrality measure.

        Parameters:
        - graph (pgl.Graph): Input graph.

        Returns:
        - paddle.Tensor: Node importance values.
        """
        # if graph.indegree().count_nonzero(axis=0) != graph.num_nodes:
        if np.count_nonzero(graph.indegree(), axis=0) != graph.num_nodes:
            graph_temp = add_self_loops(graph.numpy())
        else:
            graph_temp = graph

        if self.centrality_measure == 'degree':
            drop_weights = degree_drop_weights(graph_temp)
        elif self.centrality_measure == 'pr':
            drop_weights = pr_drop_weights(graph_temp, aggr='sink', k=200)
        elif self.centrality_measure == 'evc':
            drop_weights = evc_drop_weights(graph_temp)
        else:
            raise Exception("Centrality measure option '{}' is not available!".format(self.centrality_measure))
        return drop_weights

    def do_augment(self, graph):
        """
        Perform the edge removal on the graph.

        Parameters:
        - graph (pgl.Graph): Input graph from which edges will be removed.

        Returns:
        - pgl.Graph: Graph with edges removed.
        """
        drop_weights = self.__get_edge_weights(graph)
        new_edges = drop_edge_weighted(graph.edges, drop_weights, p=self.delete_ratio, threshold=self.threshold)
        new_graph = pgl.Graph(num_nodes=graph.num_nodes, edges=new_edges, node_feat=graph.node_feat,
                              edge_feat=graph.edge_feat)
        return new_graph

    def views_fn(self, graph):
        """
        Wrapper function to call the edge removal function.

        Parameters:
        - graph (pgl.Graph): Input graph from which edges will be removed.

        Returns:
        - pgl.Graph: Graph with edges removed.
        """
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)


class GraphDiffusion():
    def __init__(self, mode='ppr', alpha=0.2, t=5, add_self_loop=True):
        self.mode = mode
        self.alpha = alpha
        self.t = t
        self.add_self_loop = add_self_loop

    def __call__(self, graph):
        return self.views_fn(graph)

    @staticmethod
    def create_adjacency_matrix(num_nodes, edges):
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for src, dst in edges:
            adj_matrix[src][dst] = 1
            adj_matrix[dst][src] = 1  # Assuming undirected graph
        return adj_matrix

    def do_trans(self, graph):
        if self.add_self_loop:
            graph = add_self_loops(graph)

        dense_adj = self.create_adjacency_matrix(graph.num_nodes, graph.edges)

        degree_matrix = np.diag(np.sum(dense_adj, axis=1))

        if self.mode == 'ppr':
            dinv = paddle.inverse(paddle.sqrt(paddle.to_tensor(degree_matrix)))
            at = paddle.matmul(paddle.matmul(dinv, paddle.to_tensor(dense_adj)), dinv)
            diff_adj = self.alpha * paddle.inverse(paddle.eye(dense_adj.shape[0]) - (1 - self.alpha) * at)
        elif self.mode == 'heat':
            d_inv = paddle.inverse(paddle.to_tensor(degree_matrix))
            diff_adj = paddle.exp(self.t * (paddle.matmul(paddle.to_tensor(dense_adj), d_inv) - paddle.eye(dense_adj.shape[0])))
        else:
            raise Exception("Invalid diffusion mode. Choose 'ppr' or 'heat'.")

        diff_adj = diff_adj.numpy()
        diff_edges = np.transpose(np.nonzero(diff_adj))
        diff_edges_tensor = paddle.to_tensor(diff_edges, dtype='int64')
        return pgl.Graph(num_nodes=graph.num_nodes, edges=diff_edges_tensor)

    def views_fn(self, graph):
        if isinstance(graph, list):
            return [self.do_trans(g) for g in graph]
        elif isinstance(graph, pgl.Graph):
            return self.do_trans(graph)

def create_test_graph():
    num_nodes = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # Simple cycle
    graph = pgl.Graph(num_nodes=num_nodes, edges=edges)
    return graph

def test_diffusion():
    graph = create_test_graph()

    ppr_diffusion = GraphDiffusion(mode='ppr', alpha=0.2, add_self_loop=True)
    ppr_diff_graph = ppr_diffusion(graph)
    print("PPR Diffused Graph:")
    print(ppr_diff_graph)

    heat_diffusion = GraphDiffusion(mode='heat', t=5, add_self_loop=True)
    heat_diff_graph = heat_diffusion(graph)
    print("\nHeat Kernel Diffused Graph:")
    print(heat_diff_graph)

test_diffusion()


def floyd_warshall(adj_matrix):
    """
    Compute the shortest path distance matrix using the Floyd-Warshall algorithm.

    Parameters:
    - adj_matrix (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
    - numpy.ndarray: Distance matrix.
    """
    num_nodes = adj_matrix.shape[0]

    # Initialize the distance matrix
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    # If there's an edge between nodes, set the distance as 1
    distance_matrix[adj_matrix > 0] = 1

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                distance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    return distance_matrix


def softmax(matrix):
    """
    Compute the softmax of a matrix.

    Parameters:
    - matrix (numpy.ndarray): Input matrix.

    Returns:
    - numpy.ndarray: Softmax of the input matrix.
    """
    e_matrix = np.exp(matrix - np.max(matrix))
    return e_matrix / e_matrix.sum(axis=1, keepdims=True)


def compute_normalized_distance(graph):
    """
    Compute the normalized distance matrix for a pgl graph.

    Parameters:
    - graph (pgl.Graph): Input graph.

    Returns:
    - numpy.ndarray: Normalized distance matrix.
    """
    # Convert the pgl graph to an adjacency matrix
    adj_matrix = graph.adjacency_matrix().todense().astype('float32')

    # Compute the distance matrix using the Floyd-Warshall algorithm
    distance_matrix = floyd_warshall(adj_matrix)

    # Normalize the distance matrix using the softmax function
    normalized_distance = softmax(distance_matrix)

    return normalized_distance


# Example usage:
# graph = pgl.graph(...)  # Create or load your pgl graph here
# normalized_distance_matrix = compute_normalized_distance(graph)

class DistMatrixAugment():
    def __init__(self):
        """
        Initialize the DistMatrixAugment class.
        """
        pass

    def __call__(self, graph):
        """
        Make the class callable. This returns the graph after augmentation.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        return self.views_fn(graph)

    def do_augment(self, graph):
        """
        use the distance matrix to augment the graph

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """

        normalized_distance_matrix = compute_normalized_distance(graph)
        normalized_distance_matrix = normalized_distance_matrix.astype('float32')
        graph.node_feat["dist"] = normalized_distance_matrix
        return graph

    def views_fn(self, graph):
        """
        Wrapper function to call the augmentation function.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)


class PGLEdgeAugmenter():
    def __init__(self):
        pass

    def graphsage_sample(self, graph, nodes, samples, ignore_edges=[]):
        """
        Implement the GraphSAGE sampling using PGL's method.

        Parameters:
        - graph (pgl.Graph): A PGL graph instance.
        - nodes: Sample starting from nodes.
        - samples: A list, number of neighbors in each layer.
        - ignore_edges: List of edge(src, dst) that will be ignored.

        Returns:
        - List of subgraphs.
        """
        return pgl.sampling.graphsage_sample(graph, nodes, samples, ignore_edges)

    def random_walk(self, graph, nodes, max_depth):
        """
        Implement the random walk sampling using PGL's method.

        Parameters:
        - graph (pgl.Graph): A PGL graph instance.
        - nodes: Walk starting from nodes.
        - max_depth: Max walking depth.

        Returns:
        - List of walks.
        """
        return pgl.sampling.random_walk(graph, nodes, max_depth)

    def subgraph(self, graph, node_ids, eid=None, edges=None, with_node_feat=True, with_edge_feat=True):
        """
        Implement the subgraph sampling using PGL's method.

        Parameters:
        - graph (pgl.Graph): A PGL graph instance.
        - node_ids: Node IDs to be included in the subgraph.
        - eid (optional): Edge IDs to be included in the subgraph.
        - edges: Edge(src, dst) list to be included in the subgraph.
        - with_node_feat: Whether to inherit node features from the parent graph.
        - with_edge_feat: Whether to inherit edge features from the parent graph.

        Returns:
        - pgl.Graph object.
        """
        return pgl.sampling.subgraph(graph, node_ids, eid, edges, with_node_feat, with_edge_feat)


class RWSample():
    def __init__(self, max_depth):
        """
        Initialize the RWSample class.

        Parameters:
        - max_depth (int): Maximum depth of the random walk.
        """
        self.max_depth = max_depth

    def __call__(self, graph):
        """
        Make the class callable. This returns the graph after augmentation.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        return self.views_fn(graph)

    def do_augment(self, graph):
        """
        Augment the graph using random walk sampling.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        # Sample random walks
        walks = pgl.sampling.random_walk(graph, list(range(graph.num_nodes)), self.max_depth)

        # Create a new graph with the random walks as edges
        src, dst = np.array(walks).T
        walk_edges = np.stack([src, dst], axis=1)
        walk_graph = pgl.Graph(num_nodes=graph.num_nodes, edges=walk_edges, node_feat=graph.node_feat,
                               edge_feat=graph.edge_feat)

        return walk_graph

    def views_fn(self, graph):
        """
        Wrapper function to call the augmentation function.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)


class SubGraph():
    def __init__(self, num_nodes, eid=None, edges=None, with_node_feat=True, with_edge_feat=True):
        """
        Initialize the SubGraph class.

        Parameters:
        - num_nodes (int): Number of nodes to be included in the subgraph.
        """
        self.num_nodes = num_nodes
        self.eid = eid
        self.edges = edges
        self.with_node_feat = with_node_feat
        self.with_edge_feat = with_edge_feat

    def __call__(self, graph):
        """
        Make the class callable. This returns the graph after augmentation.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        return self.views_fn(graph)

    def do_augment(self, graph):
        """
        Augment the graph using subgraph sampling.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        subgraph = pgl.sampling.subgraph(graph, list(range(self.num_nodes)), self.eid, self.edges, self.with_node_feat,
                                         self.with_edge_feat)

        return subgraph

    def views_fn(self, graph):
        """
        Wrapper function to call the augmentation function.

        Parameters:
        - graph (pgl.Graph): Input graph to be augmented.

        Returns:
        - pgl.Graph: Augmented graph.
        """
        if isinstance(graph, list):
            return [self.do_augment(g) for g in graph]
        elif isinstance(graph, Graph):
            return self.do_augment(graph)
