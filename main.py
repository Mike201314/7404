import click
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import TabularDataLoader
from incremental_learner import IncrementalLearner
from utils import Timer, Statistics
from visualizer import TreeVisualizer


@click.group()
def cli():
    """Optimal Decision Tree Inference Tool"""
    pass


@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('-d', '--depth-only', is_flag=True, help='Minimize only depth')
@click.option('-k', '--min-depth', default=1, help='Minimum depth')
@click.option('-n', '--max-nodes', type=int, help='Maximum number of nodes (uses Algorithm 3)')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
@click.option('-t', '--test-file', type=click.Path(exists=True), help='Test dataset')
def infer(csv_file, depth_only, min_depth, max_nodes, verbose, test_file):
    """Infer optimal decision tree"""

    # Load data
    loader = TabularDataLoader(csv_file, test_file)
    train_data = loader.get_binarized_training()

    # Initialize learner
    learner = IncrementalLearner(train_data, verbose=verbose)

    # Find optimal tree
    timer = Timer("Tree Inference")

    if max_nodes is not None:
        # Use Algorithm 3 with MaxNodes constraint
        if verbose:
            click.echo(f"Using Algorithm 3 with MaxNodes={max_nodes}")
        tree = learner.find_optimal_tree(min_depth, minimize_nodes=False, max_nodes=max_nodes)
    else:
        # Original approach
        tree = learner.find_optimal_tree(min_depth, not depth_only)

    inference_time = timer.stop()

    if tree is None:
        click.echo("No solution found!")
        return

    # Print results
    if verbose:
        click.echo(f"\n===== RESULTS =====")
        click.echo(f"Depth: {tree.depth}")
        click.echo(f"Nodes: {tree.num_nodes}")
        click.echo(f"Time: {inference_time:.3f}s")
        if max_nodes is not None:
            click.echo(f"MaxNodes constraint: {max_nodes} (satisfied: {tree.num_nodes <= max_nodes})")

    # Test accuracy
    if test_file:
        test_data = loader.get_binarized_testing()
        accuracy = tree.evaluate(test_data)
        click.echo(f"Accuracy: {accuracy:.1%}")

    # Visualize
    visualizer = TreeVisualizer(tree, train_data)
    click.echo("\nTree structure:")
    click.echo(visualizer.to_text())


@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('-f', '--folds', default=5, help='Cross-validation folds')
@click.option('-n', '--max-nodes', type=int, help='Maximum number of nodes')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
def bench(csv_file, folds, max_nodes, verbose):
    """Run cross-validation benchmark"""

    loader = TabularDataLoader(csv_file)
    data = loader.get_binarized_training()

    stats_accuracy = Statistics("Accuracy (%)")
    stats_time = Statistics("Time (ms)")
    stats_depth = Statistics("Depth")
    stats_nodes = Statistics("Nodes")

    for fold in range(folds):
        if verbose:
            click.echo(f"Fold {fold + 1}/{folds}")

        # Split data
        train_data, test_data = data.cross_validation_split(folds, fold)

        # Train
        learner = IncrementalLearner(train_data, verbose=False)
        timer = Timer()

        if max_nodes is not None:
            tree = learner.find_optimal_tree(1, minimize_nodes=False, max_nodes=max_nodes)
        else:
            tree = learner.find_optimal_tree(1, True)

        time_ms = timer.stop() * 1000

        if tree:
            accuracy = tree.evaluate(test_data) * 100
            stats_accuracy.add(accuracy)
            stats_time.add(time_ms)
            stats_depth.add(tree.depth)
            stats_nodes.add(tree.num_nodes)

    # Print results
    constraint_info = f" (MaxNodes={max_nodes})" if max_nodes else ""
    click.echo(f"\n===== RESULTS ({folds}-fold CV){constraint_info} =====")
    click.echo(f"Accuracy: {stats_accuracy.mean:.1f}% ± {stats_accuracy.std:.1f}%")
    click.echo(f"Depth: {stats_depth.mean:.1f} ± {stats_depth.std:.1f}")
    click.echo(f"Nodes: {stats_nodes.mean:.1f} ± {stats_nodes.std:.1f}")
    click.echo(f"Time: {stats_time.mean:.1f} ± {stats_time.std:.1f} ms")


@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('-k', '--depth', default=3, help='Tree depth')
@click.option('-n', '--max-nodes', type=int, help='Maximum number of nodes')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
def algorithm3(csv_file, depth, max_nodes, verbose):
    """Run Algorithm 3 directly with specified depth and max nodes"""

    loader = TabularDataLoader(csv_file)
    data = loader.get_binarized_training()

    learner = IncrementalLearner(data, verbose=verbose)

    click.echo(f"Running Algorithm 3 with depth={depth}, max_nodes={max_nodes}")

    timer = Timer()
    tree = learner.infer_decision_tree(depth, max_nodes)
    time_taken = timer.stop()

    if tree:
        accuracy = tree.evaluate(data)
        click.echo(f"\n===== ALGORITHM 3 RESULTS =====")
        click.echo(f"Depth: {tree.depth}")
        click.echo(f"Nodes: {tree.num_nodes}")
        click.echo(f"Accuracy: {accuracy:.1%}")
        click.echo(f"Time: {time_taken:.3f}s")

        if max_nodes is not None:
            satisfied = tree.num_nodes <= max_nodes
            click.echo(f"MaxNodes constraint: {max_nodes} (satisfied: {satisfied})")

        # Show tree
        visualizer = TreeVisualizer(tree, data)
        click.echo("\nTree structure:")
        click.echo(visualizer.to_text())
    else:
        click.echo("No solution found!")


@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('-r', '--runs', default=20, help='Number of runs for averaging (default: 20)')
@click.option('-f', '--folds', default=10, help='Cross-validation folds (default: 10)')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
def table(csv_file, runs, folds, verbose):
    """Generate paper-style benchmark table (Table 1 reproduction)"""

    sys.path.append(str(Path(__file__).parent))
    from demo import run_algorithm_benchmark, run_cross_validation_benchmark, print_benchmark_table

    loader = TabularDataLoader(csv_file)
    data = loader.get_binarized_training()

    dataset_name = Path(csv_file).stem.title()

    click.echo(f"Generating benchmark table for {dataset_name} dataset")
    click.echo(f"Runs per algorithm: {runs}")
    click.echo(f"Cross-validation folds: {folds}")
    click.echo()

    # Define algorithms to test
    algorithms = [
        ("Our_DT_size", {"min_depth": 1, "minimize_nodes": True}),
        ("Our_DT_depth", {"min_depth": 1, "minimize_nodes": False}),
        ("Our_DT_max10", {"min_depth": 1, "max_nodes": 10}),
        ("Our_DT_max15", {"min_depth": 1, "max_nodes": 15}),
    ]

    # Run training set benchmarks
    results = []
    for algo_name, params in algorithms:
        if not verbose:
            click.echo(f"Running {algo_name}...")
        result = run_algorithm_benchmark(data, algo_name, runs, **params)
        results.append(result)

    # Run cross-validation benchmarks
    cv_results = []
    for algo_name, params in algorithms:
        if not verbose:
            click.echo(f"Running {algo_name} cross-validation...")
        cv_result = run_cross_validation_benchmark(data, f"{algo_name}_CV", folds, **params)
        cv_results.append(cv_result)

    # Print results in paper format
    print_benchmark_table(results, cv_results, dataset_name)


if __name__ == '__main__':
    cli()