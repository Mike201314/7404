import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data_loader import TabularDataLoader
from src.incremental_learner import IncrementalLearner
from src.utils import Timer, Statistics


def benchmark_dataset(csv_file: str, name: str, n_folds: int = 5):
    """Benchmark single dataset"""
    print(f"\n=== Benchmarking {name} ===")

    if not Path(csv_file).exists():
        print(f"Dataset {csv_file} not found")
        return

    loader = TabularDataLoader(csv_file)
    data = loader.get_binarized_training()

    print(f"Classes: {data.num_classes}, Features: {data.num_features}")
    print(f"Examples: {sum(len(class_data) for class_data in data.data)}")

    stats_accuracy = Statistics("Accuracy (%)")
    stats_time = Statistics("Time (ms)")
    stats_depth = Statistics("Depth")
    stats_nodes = Statistics("Nodes")

    for fold in range(n_folds):
        print(f"  Fold {fold + 1}/{n_folds}")

        train_data, test_data = data.cross_validation_split(n_folds, fold, seed=42)

        learner = IncrementalLearner(train_data, verbose=False)
        timer = Timer()
        tree = learner.find_optimal_tree(min_depth=1, minimize_nodes=True)
        time_ms = timer.stop() * 1000

        if tree:
            accuracy = tree.evaluate(test_data) * 100
            stats_accuracy.add(accuracy)
            stats_time.add(time_ms)
            stats_depth.add(tree.depth)
            stats_nodes.add(tree.num_nodes)

    # Print results
    if stats_accuracy.count > 0:
        print(f"Accuracy: {stats_accuracy.mean:.1f}% ± {stats_accuracy.std:.1f}%")
        print(f"Depth: {stats_depth.mean:.1f} ± {stats_depth.std:.1f}")
        print(f"Nodes: {stats_nodes.mean:.1f} ± {stats_nodes.std:.1f}")
        print(f"Time: {stats_time.mean:.1f} ± {stats_time.std:.1f} ms")


def main():
    """Run benchmarks"""
    datasets = [
        ("data/mouse.csv", "Mouse"),
        ("data/car.csv", "Car"),
        ("data/monks-problems/monks-1.csv", "MONKS-1"),
        ("data/monks-problems/monks-2.csv", "MONKS-2"),
        ("data/monks-problems/monks-3.csv", "MONKS-3"),
    ]

    for csv_file, name in datasets:
        benchmark_dataset(csv_file, name)


if __name__ == "__main__":
    main()