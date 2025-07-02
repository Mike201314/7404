from typing import List, Dict, Optional
import math
from dataclasses import dataclass


@dataclass
class TreeNode:
    """A node in the decision tree"""
    node_id: int
    feature_id: Optional[int] = None  # None for leaves
    class_label: Optional[int] = None  # None for internal nodes

    @property
    def is_leaf(self) -> bool:
        return self.feature_id is None


class DecisionTree:
    """Decision tree implementation"""

    def __init__(self, tree_array: List[int] = None, depth: int = None):
        self.nodes: Dict[int, TreeNode] = {}
        self.depth = depth or 0
        self.num_nodes = 0

        if tree_array is not None:
            self._build_from_array(tree_array)

    def _build_from_array(self, tree_array: List[int]):
        """Build tree from SAT solver array"""
        if not tree_array:
            return

        self.depth = int(math.log2(len(tree_array))) - 1

        for i in range(1, len(tree_array)):
            value = tree_array[i]
            if value == float('-inf') + 1:
                continue

            node = TreeNode(node_id=i)
            if value >= 0:
                node.feature_id = value
            else:
                node.class_label = -value - 1

            self.nodes[i] = node

        self.num_nodes = len([n for n in self.nodes.values()
                              if n.feature_id is not None or n.class_label is not None])

    def classify(self, features: List[bool]) -> int:
        """Classify example"""
        if not self.nodes:
            return 0

        current_id = 1
        while current_id in self.nodes:
            node = self.nodes[current_id]
            if node.is_leaf:
                return node.class_label or 0

            if features[node.feature_id]:
                current_id = current_id * 2 + 1  # Right
            else:
                current_id = current_id * 2  # Left

        return 0

    def evaluate(self, data) -> float:
        """Evaluate accuracy on dataset"""
        correct = 0
        total = 0

        for class_label, examples in enumerate(data.data):
            for example in examples:
                predicted = self.classify(example)
                if predicted == class_label:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0