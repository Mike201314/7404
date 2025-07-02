from typing import Optional, Tuple, List, Dict, Any
from sat_encoder import SATEncoder
from decision_tree import DecisionTree


class InferenceResult:
    """Result of tree inference including statistics"""

    def __init__(self, tree: DecisionTree, examples_used: int, time_taken: float, accuracy: float = None):
        self.tree = tree
        self.examples_used = examples_used
        self.time_taken = time_taken
        self.accuracy = accuracy
        self.depth = tree.depth if tree else 0
        self.nodes = tree.num_nodes if tree else 0


class IncrementalLearner:
    """Incremental decision tree learner following Algorithm 3"""

    def __init__(self, data, verbose: bool = False):
        self.data = data
        self.verbose = verbose
        self.processed_examples: set = set()

    def find_optimal_tree_with_stats(self, min_depth: int = 1, minimize_nodes: bool = False,
                                     max_nodes: Optional[int] = None) -> Optional[InferenceResult]:
        """
        Find optimal tree and return detailed statistics

        Returns:
            InferenceResult with tree, examples_used, time_taken, etc.
        """
        import time
        start_time = time.perf_counter()

        tree = self.find_optimal_tree(min_depth, minimize_nodes, max_nodes)

        end_time = time.perf_counter()
        time_taken = end_time - start_time

        if tree is None:
            return None

        examples_used = len(self.processed_examples)
        accuracy = tree.evaluate(self.data)

        return InferenceResult(tree, examples_used, time_taken, accuracy)

    def run_cross_validation(self, n_folds: int = 10, min_depth: int = 1,
                             minimize_nodes: bool = False, max_nodes: Optional[int] = None) -> Dict[str, float]:
        """
        Run cross-validation and return average statistics

        Returns:
            Dictionary with mean values for time, accuracy, depth, nodes, examples_used
        """
        import time

        times = []
        accuracies = []
        depths = []
        nodes = []
        examples_used = []

        for fold in range(n_folds):
            train_data, test_data = self.data.cross_validation_split(n_folds, fold, seed=42)

            # Create new learner for this fold
            fold_learner = IncrementalLearner(train_data, verbose=False)

            start_time = time.perf_counter()
            tree = fold_learner.find_optimal_tree(min_depth, minimize_nodes, max_nodes)
            end_time = time.perf_counter()

            if tree:
                times.append(end_time - start_time)
                accuracies.append(tree.evaluate(test_data) * 100)  # Convert to percentage
                depths.append(tree.depth)
                nodes.append(tree.num_nodes)
                examples_used.append(len(fold_learner.processed_examples))

        return {
            'time': sum(times) / len(times) if times else 0,
            'accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'depth': sum(depths) / len(depths) if depths else 0,
            'nodes': sum(nodes) / len(nodes) if nodes else 0,
            'examples_used': sum(examples_used) / len(examples_used) if examples_used else 0
        }

    def find_optimal_tree(self, min_depth: int = 1, minimize_nodes: bool = False,
                          max_nodes: Optional[int] = None) -> Optional[DecisionTree]:
        """
        Find optimal decision tree

        Args:
            min_depth: Minimum depth to start searching
            minimize_nodes: Whether to minimize nodes using binary search
            max_nodes: Maximum number of nodes (if specified, uses Algorithm 3 directly)
        """
        # If max_nodes is specified, use Algorithm 3 directly
        if max_nodes is not None:
            for depth in range(min_depth, 10):
                if self.verbose:
                    print(f"Trying depth k={depth} with MaxNodes={max_nodes}")

                # Convert nodes to leaves constraint (nodes = 2*leaves - 1)
                max_leaves = (max_nodes + 1) // 2
                tree = self._infer_tree_algorithm3(depth, max_leaves)
                if tree is not None:
                    return tree
            return None

        # Original approach: find minimal depth first
        for depth in range(min_depth, 10):
            if self.verbose:
                print(f"Trying depth k={depth}")

            tree = self._infer_tree(depth)
            if tree is not None:
                if not minimize_nodes:
                    return tree
                return self._minimize_nodes(depth, tree)

        return None

    def infer_decision_tree(self, depth: int, max_nodes: Optional[int] = None) -> Optional[DecisionTree]:
        """
        Algorithm 3: InferDecisionTree

        Args:
            depth: Maximum depth k of the tree
            max_nodes: Maximum number of nodes (optional)

        Returns:
            Decision tree consistent with training examples, or None if no solution
        """
        if max_nodes is not None:
            # Convert nodes to leaves constraint
            max_leaves = (max_nodes + 1) // 2
            return self._infer_tree_algorithm3(depth, max_leaves)
        else:
            return self._infer_tree(depth)

    def _infer_tree_algorithm3(self, depth: int, max_leaves: Optional[int] = None) -> Optional[DecisionTree]:
        """
        Complete Algorithm 3 implementation with MaxNodes support

        This follows the exact algorithm from the paper:
        1: C ← formulas (1) and (2)
        2: while C is satisfiable do
        3:   Let T be a decision tree of a solution of C
        4:   if E ⊆ T then
        5:     return T
        6:   end if
        7:   Let e ∈ Ea be an example mislabeled by T
        8:   C ← C ∧ GenerateFeatureConstraints(e,∅,1,0) ∧ GenerateClassConstraints(e,∅,0,0,k,a)
        9:   if MaxNodes is defined then
        10:    C ← C ∧ C' where C' is constraints (7), (8), (9) and (10)
        11:  end if
        12: end while
        13: return "No solution"
        """
        # Step 1: Initialize with structural constraints (formulas 1 and 2)
        encoder = SATEncoder(depth, self.data.num_features, self.data.num_classes)
        self.processed_examples = set()

        # Add initial node counting constraints if MaxNodes is defined
        if max_leaves is not None:
            encoder.add_node_counting_constraints(max_leaves)
            if self.verbose:
                print(f"  Added initial node counting constraints (max_leaves={max_leaves})")

        iteration = 0
        # Step 2: Main loop - while C is satisfiable
        while True:
            iteration += 1
            if self.verbose:
                print(f"  Iteration {iteration}: Solving SAT with {len(encoder.cnf.clauses)} clauses")

            # Step 3: Get decision tree from SAT solution
            tree_array = encoder.solve(max_leaves)
            if tree_array is None:
                # Step 13: No solution
                if self.verbose:
                    print("  SAT formula unsatisfiable - no solution")
                return None

            tree = DecisionTree(tree_array, depth)

            # Step 4: Check if E ⊆ T (all examples correctly classified)
            misclassified = self._find_misclassified_example(tree)
            if misclassified is None:
                # Step 5: Return T
                if self.verbose:
                    print(f"  Found consistent tree after {iteration} iterations")
                    print(f"  Processed {len(self.processed_examples)} examples")
                return tree

            # Step 7: Get misclassified example e ∈ Ea
            example_idx, features, class_label = misclassified
            self.processed_examples.add(example_idx)

            if self.verbose:
                print(f"  Found misclassified example {len(self.processed_examples)} (class {class_label})")

            # Step 8: Add constraints for the misclassified example
            # C ← C ∧ GenerateFeatureConstraints(e,∅,1,0) ∧ GenerateClassConstraints(e,∅,0,0,k,a)
            encoder.add_example(features, class_label)

            # Steps 9-11: If MaxNodes is defined, add node counting constraints
            # Note: We already added initial constraints, and the encoder maintains them
            # across example additions, so no additional action needed here

    def _infer_tree(self, depth: int) -> Optional[DecisionTree]:
        """Original incremental approach without MaxNodes constraint"""
        encoder = SATEncoder(depth, self.data.num_features, self.data.num_classes)
        self.processed_examples = set()

        while True:
            tree_array = encoder.solve()
            if tree_array is None:
                return None

            tree = DecisionTree(tree_array, depth)
            misclassified = self._find_misclassified_example(tree)

            if misclassified is None:
                return tree

            example_idx, features, label = misclassified
            encoder.add_example(features, label)
            self.processed_examples.add(example_idx)

            if self.verbose:
                print(f"  Added example {len(self.processed_examples)}")

    def _minimize_nodes(self, depth: int, initial_tree: DecisionTree) -> DecisionTree:
        """Minimize nodes using binary search"""
        left, right = 1, 2 ** depth
        best_tree = initial_tree

        while left <= right:
            mid = (left + right) // 2

            if self.verbose:
                print(f"  Trying MaxNodes={2 * mid - 1}")

            # Use Algorithm 3 with max_leaves constraint
            tree = self._infer_tree_algorithm3(depth, mid)

            if tree is not None:
                best_tree = tree
                right = mid - 1
                if self.verbose:
                    print("    Solution found")
            else:
                left = mid + 1
                if self.verbose:
                    print("    No solution")

        return best_tree

    def _find_misclassified_example(self, tree: DecisionTree) -> Optional[Tuple[int, List[bool], int]]:
        """Find misclassified example"""
        example_idx = 0

        for class_label, examples in enumerate(self.data.data):
            for features in examples:
                if example_idx not in self.processed_examples:
                    predicted = tree.classify(features)
                    if predicted != class_label:
                        return (example_idx, features, class_label)
                example_idx += 1

        return None

    def _get_example(self, example_idx: int) -> Tuple[int, List[bool]]:
        """Get example by index"""
        current_idx = 0
        for class_label, examples in enumerate(self.data.data):
            for features in examples:
                if current_idx == example_idx:
                    return (class_label, features)
                current_idx += 1
        raise IndexError(f"Example {example_idx} not found")
