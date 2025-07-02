from typing import List, Dict, Optional, Tuple
import math
from pysat.solvers import Minisat22
from pysat.formula import CNF


class SATEncoder:
    """SAT encoder for decision tree inference"""

    def __init__(self, depth: int, num_features: int, num_classes: int):
        self.depth = depth
        self.num_features = num_features
        self.num_classes = num_classes

        # Variable management
        self.var_counter = 1
        self.X_vars: Dict[Tuple[int, int], int] = {}
        self.F_vars: Dict[Tuple[int, int], int] = {}
        self.C_vars: Dict[Tuple[int, int], int] = {}
        self.U_vars: Dict[int, int] = {}
        self.H_vars: Dict[Tuple[int, int], int] = {}

        # Examples
        self.num_examples = 0
        self.examples: List[List[bool]] = []
        self.example_labels: List[int] = []

        # SAT formula
        self.cnf = CNF()
        self._add_structural_constraints()

    def _get_variable(self, var_dict: Dict, key) -> int:
        """Get or create SAT variable"""
        if key not in var_dict:
            var_dict[key] = self.var_counter
            self.var_counter += 1
        return var_dict[key]

    def get_X_var(self, example_id: int, level: int) -> int:
        return self._get_variable(self.X_vars, (example_id, level))

    def get_F_var(self, node_id: int, feature_id: int) -> int:
        return self._get_variable(self.F_vars, (node_id, feature_id))

    def get_C_var(self, leaf_id: int, class_id: int) -> int:
        return self._get_variable(self.C_vars, (leaf_id, class_id))

    def get_U_var(self, leaf_id: int) -> int:
        return self._get_variable(self.U_vars, leaf_id)

    def get_H_var(self, i: int, j: int) -> int:
        return self._get_variable(self.H_vars, (i, j))

    def _add_structural_constraints(self):
        """Add structural constraints"""
        num_internal_nodes = 2 ** self.depth - 1

        for node_id in range(1, num_internal_nodes + 1):
            # Each node has exactly one feature
            clause = [self.get_F_var(node_id, f) for f in range(self.num_features)]
            self.cnf.append(clause)

            for f1 in range(self.num_features):
                for f2 in range(f1 + 1, self.num_features):
                    self.cnf.append([
                        -self.get_F_var(node_id, f1),
                        -self.get_F_var(node_id, f2)
                    ])

    def add_example(self, features: List[bool], class_label: int):
        """Add training example"""
        example_id = self.num_examples
        self.examples.append(features)
        self.example_labels.append(class_label)
        self.num_examples += 1

        self._generate_feature_constraints(example_id, features, [], 1, 0)
        self._generate_class_constraints(example_id, class_label, [], 0, 0)

    def _generate_feature_constraints(self, example_id: int, features: List[bool],
                                      clause: List[int], node_id: int, level: int):
        """Generate feature constraints (Algorithm 1)"""
        if level == self.depth:
            return

        for feature_id in range(self.num_features):
            if not features[feature_id]:
                constraint = clause + [
                    -self.get_X_var(example_id, level),
                    -self.get_F_var(node_id, feature_id)
                ]
                self.cnf.append(constraint)

        left_clause = clause + [-self.get_X_var(example_id, level)]
        self._generate_feature_constraints(example_id, features, left_clause,
                                           node_id * 2 + 1, level + 1)

        for feature_id in range(self.num_features):
            if features[feature_id]:
                constraint = clause + [
                    self.get_X_var(example_id, level),
                    -self.get_F_var(node_id, feature_id)
                ]
                self.cnf.append(constraint)

        right_clause = clause + [self.get_X_var(example_id, level)]
        self._generate_feature_constraints(example_id, features, right_clause,
                                           node_id * 2, level + 1)

    def _generate_class_constraints(self, example_id: int, class_label: int,
                                    clause: List[int], leaf_id: int, level: int):
        """Generate class constraints (Algorithm 2)"""
        if level == self.depth:
            constraint = clause + [self.get_C_var(leaf_id, class_label)]
            self.cnf.append(constraint)

            for other_class in range(self.num_classes):
                if other_class != class_label:
                    constraint = clause + [-self.get_C_var(leaf_id, other_class)]
                    self.cnf.append(constraint)
            return

        left_clause = clause + [self.get_X_var(example_id, level)]
        self._generate_class_constraints(example_id, class_label, left_clause,
                                         leaf_id * 2, level + 1)

        right_clause = clause + [-self.get_X_var(example_id, level)]
        self._generate_class_constraints(example_id, class_label, right_clause,
                                         leaf_id * 2 + 1, level + 1)

    def add_node_counting_constraints(self, max_leaves: int):
        """Add node counting constraints"""
        num_leaves = 2 ** self.depth

        for leaf_id in range(num_leaves):
            for class_id in range(self.num_classes):
                self.cnf.append([
                    -self.get_C_var(leaf_id, class_id),
                    self.get_U_var(leaf_id)
                ])

        for i in range(num_leaves):
            for j in range(max_leaves + 1):
                if i > 0:
                    self.cnf.append([
                        -self.get_H_var(i, j),
                        self.get_H_var(i + 1, j)
                    ])

                if j < max_leaves:
                    self.cnf.append([
                        -self.get_U_var(i),
                        -self.get_H_var(i, j),
                        self.get_H_var(i + 1, j + 1)
                    ])

        self.cnf.append([self.get_H_var(0, 0)])
        self.cnf.append([-self.get_H_var(num_leaves, max_leaves + 1)])

    def solve(self, max_leaves: Optional[int] = None) -> Optional[List[int]]:
        """Solve SAT formula"""
        solver = Minisat22()
        solver.append_formula(self.cnf)

        if max_leaves is not None:
            solver.add_clause([-self.get_H_var(2 ** self.depth, max_leaves + 1)])

        if solver.solve():
            model = solver.get_model()
            tree = self._extract_tree(model)
            solver.delete()
            return tree

        solver.delete()
        return None

    def _extract_tree(self, model: List[int]) -> List[int]:
        """Extract tree from SAT model"""
        model_set = set(model)
        tree_size = 2 ** (self.depth + 1)
        tree = [float('-inf')] * tree_size

        num_internal_nodes = 2 ** self.depth - 1
        for node_id in range(1, num_internal_nodes + 1):
            for feature_id in range(self.num_features):
                if self.get_F_var(node_id, feature_id) in model_set:
                    tree[node_id] = feature_id
                    break

        num_leaves = 2 ** self.depth
        for leaf_id in range(num_leaves):
            for class_id in range(self.num_classes):
                if self.get_C_var(leaf_id, class_id) in model_set:
                    tree[2 ** self.depth + leaf_id] = -(class_id + 1)
                    break

        return tree