import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
import time
from src.data_loader import TabularDataLoader, BinarizedData
from src.incremental_learner import IncrementalLearner, InferenceResult
from src.visualizer import TreeVisualizer
from src.utils import Timer


def create_sample_data():
    """Create synthetic Boolean dataset similar to Mouse"""
    np.random.seed(42)

    n_samples = 70  # Same as Mouse dataset
    n_features = 45  # Same as Mouse dataset
    X = np.random.randint(0, 2, size=(n_samples, n_features))

    # More complex Boolean function to simulate Mouse complexity
    y1 = (X[:, 0] & X[:, 1]) | (X[:, 2] & ~X[:, 3])
    y2 = (X[:, 4] & X[:, 5] & X[:, 6]) | (~X[:, 7] & X[:, 8])
    y3 = X[:, 9] ^ X[:, 10]
    y4 = (X[:, 11] | X[:, 12]) & (~X[:, 13] | X[:, 14])

    y = (y1 | y2) ^ (y3 & y4)
    y = y.astype(int)

    # Convert to BinarizedData
    data = BinarizedData()
    for label in [0, 1]:
        mask = y == label
        examples = X[mask].astype(bool).tolist()
        data.add_class_data(label, examples)

    data.set_feature_names([f"feature_{i:02d}" for i in range(n_features)])
    data.set_class_names(["Class_0", "Class_1"])

    return data


def demo_csv_data():
    """Load CSV data if available"""
    csv_file = Path("data/monk1.csv")
    if csv_file.exists():
        print(f"Loading {csv_file}")
        loader = TabularDataLoader(str(csv_file))
        return loader.get_binarized_training(), "Mouse"
    else:
        print("CSV data not found, using synthetic Mouse-like data")
        return create_sample_data(), "Synthetic Mouse"


def run_algorithm_benchmark(data, algorithm_name: str, n_runs: int = 20, **kwargs) -> dict:
    """
    Run algorithm multiple times and return average statistics

    Args:
        data: Training data
        algorithm_name: Name for display
        n_runs: Number of runs for averaging
        **kwargs: Parameters for find_optimal_tree
    """
    print(f"Running {algorithm_name} ({n_runs} runs)...")

    times = []
    examples_used = []
    depths = []
    nodes = []
    accuracies = []

    for run in range(n_runs):
        learner = IncrementalLearner(data, verbose=False)

        start_time = time.perf_counter()
        tree = learner.find_optimal_tree(**kwargs)
        end_time = time.perf_counter()

        if tree:
            times.append(end_time - start_time)
            examples_used.append(len(learner.processed_examples))
            depths.append(tree.depth)
            nodes.append(tree.num_nodes)
            accuracies.append(tree.evaluate(data) * 100)  # Convert to percentage

        if (run + 1) % 5 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")

    if not times:
        return None

    return {
        'algo': algorithm_name,
        'time': np.mean(times),
        'time_std': np.std(times),
        'expl': int(np.mean(examples_used)),
        'expl_std': np.std(examples_used),
        'depth': np.mean(depths),
        'nodes': int(np.mean(nodes)),
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies)
    }


def run_cross_validation_benchmark(data, algorithm_name: str, n_folds: int = 10, **kwargs) -> dict:
    """Run cross-validation benchmark"""
    print(f"Running {algorithm_name} cross-validation ({n_folds} folds)...")

    learner = IncrementalLearner(data, verbose=False)
    cv_results = learner.run_cross_validation(n_folds, **kwargs)

    return {
        'algo': algorithm_name,
        'cv_accuracy': cv_results['accuracy'],
        'cv_time': cv_results['time'],
        'cv_expl': int(cv_results['examples_used']),
        'cv_depth': cv_results['depth'],
        'cv_nodes': int(cv_results['nodes'])
    }


def print_benchmark_table(results, cv_results, dataset_name):
    """Print paper-style benchmark table"""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS FOR '{dataset_name.upper()}' DATASET")
    print(f"{'=' * 80}")

    # Main benchmark table (similar to Table 1 in paper)
    print("\nTable: Benchmark Results (Training Set)")
    print(f"{'Algo':<12} {'Time (s)':<10} {'expl':<6} {'k':<4} {'n':<4} {'acc.':<8}")
    print("-" * 50)

    for result in results:
        if result:
            print(f"{result['algo']:<12} {result['time']:<10.3f} {result['expl']:<6} "
                  f"{result['depth']:<4.1f} {result['nodes']:<4} {result['accuracy']:<8.1f}%")

    # Cross-validation results
    print(f"\nTable: 10-fold Cross-Validation Results")
    print(f"{'Algo':<12} {'CV Acc.':<10} {'CV Time (s)':<12} {'CV expl':<8} {'CV k':<6} {'CV n':<6}")
    print("-" * 60)

    for result in cv_results:
        if result:
            print(f"{result['algo']:<12} {result['cv_accuracy']:<10.1f}% {result['cv_time']:<12.3f} "
                  f"{result['cv_expl']:<8} {result['cv_depth']:<6.1f} {result['cv_nodes']:<6}")


def show_detailed_statistics(results):
    print(f"{'=' * 60}")

    for result in results:
        if result:
            print(f"\n{result['algo']}:")
            print(f"  Time (s):    {result['time']:.3f} ± {result['time_std']:.3f}")
            print(f"  Examples:    {result['expl']} ± {result['expl_std']:.1f}")
            print(f"  Accuracy (%): {result['accuracy']:.1f} ± {result['accuracy_std']:.1f}")


def main():
    print("=== Paper-Style Benchmark Demo ===")
    print("Reproducing Table 1 from Avellaneda's paper\n")

    data, dataset_name = demo_csv_data()

    print(f"Dataset: {dataset_name}")
    print(f"Classes: {data.num_classes}, Features: {data.num_features}")
    print(f"Examples: {sum(len(cd) for cd in data.data)}\n")


    n_runs = 100

    algorithms = [
        ("Our_DT_size", {"min_depth": 1, "minimize_nodes": True}),
        ("Our_DT_depth", {"min_depth": 1, "minimize_nodes": False}),
    ]

    results = []
    for algo_name, params in algorithms:
        result = run_algorithm_benchmark(data, algo_name, n_runs, **params)
        results.append(result)

    cv_results = []
    for algo_name, params in algorithms:
        cv_result = run_cross_validation_benchmark(data, f"{algo_name}_CV", **params)
        cv_results.append(cv_result)

    # Print results in paper format
    print_benchmark_table(results, cv_results, dataset_name)

    # Show detailed statistics
    show_detailed_statistics(results)

    # Show best tree visualization
    print(f"\n{'=' * 60}")
    print("BEST TREE VISUALIZATION")
    print(f"{'=' * 60}")

    # Find best result
    valid_results = [r for r in results if r is not None]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['time'])
        print(f"Showing tree from fastest algorithm: {best_result['algo']}")

        # Re-run best algorithm to get tree for visualization
        learner = IncrementalLearner(data, verbose=False)
        best_algo_params = next(params for name, params in algorithms if name == best_result['algo'])
        tree = learner.find_optimal_tree(**best_algo_params)

        if tree:
            visualizer = TreeVisualizer(tree, data)
            print(visualizer.to_text())

            # Try graphical visualization
            try:
                print("Attempting to display graph...")
                visualizer.show_graph()
            except Exception as e:
                print(f"Graph display failed: {e}")


if __name__ == "__main__":
    main()
