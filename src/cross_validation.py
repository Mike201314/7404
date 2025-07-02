import time
import numpy as np
from src.data_loader import TabularData
from src.incremental_learner import IncrementalLearner
from src.decision_tree import score, count_inner_nodes, estimate_tree_depth

def run_cross_validation(file_path: str, folds: int = 10, min_k: int = 1,
                         max_nodes: int = None, minimize_nodes: bool = True,
                         use_equality: bool = True, verbose: bool = False):
    data = TabularData(file_path)
    full_data = data.get_binarized_data(use_equality)
    class_data = full_data.getData()
    num_classes = full_data.numClasses()

    # 将每个类按 folds 均分，并复制每折样本，防止数据泄漏
    partitions = [[] for _ in range(folds)]
    for cls_id, cls_samples in enumerate(class_data):
        cls_chunks = np.array_split(cls_samples, folds)
        for i in range(folds):
            partitions[i].append((cls_id, cls_chunks[i].copy()))  # 防止 test/train 数据共享引用

    scores, depths, nodes, times = [], [], [], []

    for i in range(folds):
        train_data = [[] for _ in range(num_classes)]
        test_data = [[] for _ in range(num_classes)]
        for fold_index, fold_partition in enumerate(partitions):
            for cls_id, samples in fold_partition:
                target = test_data if fold_index == i else train_data
                target[cls_id].extend(samples)

        BinDataClass = full_data.__class__
        bin_train = BinDataClass(train_data, full_data._bin_feature_names, full_data._class_names)

        learner = IncrementalLearner(bin_train)

        # === 节点数限制（转换 MaxNodes → MaxLeaves）===
        max_leaves = (max_nodes // 2) + 1 if max_nodes else None

        start_time = time.time()
        tree = learner.find_optimal_tree(
            min_k=min_k,
            max_nodes=max_leaves,
            minimize_nodes=minimize_nodes,
            verbose=verbose
        )
        elapsed = time.time() - start_time

        acc = score(tree, test_data)
        depth = estimate_tree_depth(tree)
        node_count = count_inner_nodes(tree)

        scores.append(acc)
        depths.append(depth)
        nodes.append(node_count)
        times.append(elapsed)

        print(f"Fold {i + 1}/{folds}: Accuracy = {acc:.2%}, Depth = {depth}, Nodes = {node_count}, Time = {elapsed:.2f}s")

    print("\n=== Cross Validation Summary ===")
    print(f"Average Accuracy: {np.mean(scores):.2%}")
    print(f"Average Depth: {np.mean(depths):.2f}")
    print(f"Average Nodes: {np.mean(nodes):.2f}")
    print(f"Average Time: {np.mean(times):.2f}s")
