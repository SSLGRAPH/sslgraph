from .node_attribute_level import *
from .topology_level import *


class Sequential():
    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(self, graph):
        return self.views_fn(graph)

    def views_fn(self, graph):
        legal_augmentation = (NodeAttrMask,
                              NodeAttrRowShuffle,
                              AdaNodeAttrMask,
                              EdgeRandomAugment,
                              AdaEdgeRemove,
                              GraphDiffusion,
                              DistMatrixAugment,
                              RWSample,
                              SubGraph)
        for augmentation in self.augmentations:
            if not isinstance(augmentation, legal_augmentation):
                raise ValueError(
                    "augmentation must be NodeAttrMask, NodeAttrRowShuffle, AdaNodeAttrMask, EdgeRandomAugment, EdgeRandomRemove, GraphDiffusion, DistMatrixAugment, RWSample or SubGraph")
            else:
                if isinstance(augmentation, (NodeAttrMask, NodeAttrRowShuffle, AdaNodeAttrMask)):
                    graph = augmentation(graph)
                elif isinstance(augmentation, (
                        EdgeRandomAugment, AdaEdgeRemove, GraphDiffusion, DistMatrixAugment, RWSample, SubGraph)):
                    new_graph = augmentation(graph)
                    graph = new_graph
        return graph
