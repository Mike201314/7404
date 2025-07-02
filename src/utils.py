import time
import math


class Timer:
    """Timer for performance measurement"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timer and return elapsed time"""
        elapsed = time.perf_counter() - self.start_time
        return elapsed


class Statistics:
    """Online statistics calculation"""

    def __init__(self, name: str = ""):
        self.name = name
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float):
        """Add value"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ==== src/visualizer.py ====
"""
Tree visualization utilities
"""
import tempfile
import subprocess
import os


class TreeVisualizer:
    """Visualize decision trees"""

    def __init__(self, tree, data):
        self.tree = tree
        self.data = data

    def to_text(self, node_id: int = 1, depth: int = 0) -> str:
        """Generate text representation"""
        if not self.tree.nodes or node_id not in self.tree.nodes:
            return "Empty tree\n"

        node = self.tree.nodes[node_id]
        indent = "| " * depth

        if node.is_leaf:
            if node.class_label is not None:
                class_name = self._get_class_name(node.class_label)
                return f"{indent}{class_name}\n"
            else:
                return f"{indent}null\n"

        feature_name = self._get_feature_name(node.feature_id)
        result = []

        # True branch (right)
        right_child = node_id * 2 + 1
        if right_child in self.tree.nodes:
            right_node = self.tree.nodes[right_child]
            if right_node.is_leaf and right_node.class_label is not None:
                class_name = self._get_class_name(right_node.class_label)
                result.append(f"{indent}{feature_name}: {class_name}\n")
            else:
                result.append(f"{indent}{feature_name}:\n")
                result.append(self.to_text(right_child, depth + 1))

        # False branch (left)
        left_child = node_id * 2
        if left_child in self.tree.nodes:
            left_node = self.tree.nodes[left_child]
            if left_node.is_leaf and left_node.class_label is not None:
                class_name = self._get_class_name(left_node.class_label)
                result.append(f"{indent}!{feature_name}: {class_name}\n")
            else:
                result.append(f"{indent}!{feature_name}:\n")
                result.append(self.to_text(left_child, depth + 1))

        return "".join(result)

    def to_dot(self) -> str:
        """Generate DOT format"""
        if not self.tree.nodes:
            return "digraph empty_tree {}"

        lines = ["digraph decision_tree {"]

        for node_id, node in self.tree.nodes.items():
            if node.is_leaf:
                if node.class_label is not None:
                    class_name = self._get_class_name(node.class_label)
                    lines.append(f'  Q{node_id} [shape="box", label="{class_name}"];')
            else:
                feature_name = self._get_feature_name(node.feature_id)
                lines.append(f'  Q{node_id} [label="{feature_name}?"];')

                left_child = node_id * 2
                right_child = node_id * 2 + 1

                if left_child in self.tree.nodes:
                    lines.append(f'  Q{node_id} -> Q{left_child} [label="no"];')
                if right_child in self.tree.nodes:
                    lines.append(f'  Q{node_id} -> Q{right_child} [label="yes"];')

        lines.append("}")
        return "\n".join(lines)

    def show_graph(self):
        """Display graph using graphviz"""
        try:
            import graphviz
            dot_source = self.to_dot()
            graph = graphviz.Source(dot_source)
            graph.render(tempfile.mktemp(), format='png', view=True, cleanup=True)
        except ImportError:
            self._show_graph_cmdline()

    def _show_graph_cmdline(self):
        """Show graph using command-line graphviz"""
        dot_source = self.to_dot()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_source)
            dot_file = f.name

        try:
            image_file = tempfile.mktemp(suffix='.png')
            subprocess.run(['dot', '-Tpng', dot_file, '-o', image_file], check=True)
            subprocess.run(['xdg-open', image_file], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Graphviz not available for visualization")
        finally:
            os.unlink(dot_file)

    def _get_feature_name(self, feature_id: int) -> str:
        if 0 <= feature_id < len(self.data.feature_names):
            return self.data.feature_names[feature_id]
        return f"feature_{feature_id}"

    def _get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.data.class_names):
            return self.data.class_names[class_id]
        return f"class_{class_id}"

