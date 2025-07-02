import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from src.data_loader import TabularDataLoader
from src.incremental_learner import IncrementalLearner
from src.utils import Timer


def compare_with_sklearn(csv_file: str):
    """Compare with sklearn decision trees"""
    print(f"Comparing algorithms on {csv_file}")

    if not Path(csv_file).exists():
        print("File not found")
        return

    # Our algorithm
    loader = TabularDataLoader(csv_file)
    data = loader.get_binarized_training()

    train_data, test_data = data.cross_validation_split(5, 0, seed=42)

    timer = Timer()
    learner = IncrementalLearner(train_data, verbose=False)
    tree = learner.find_optimal_tree(min_depth=1, minimize_nodes=True)
    our_time = timer.stop()

    our_accuracy = tree.evaluate(test_data) if tree else 0

    print(f"Our algorithm:")
    print(f"  Accuracy: {our_accuracy:.3f}")
    print(f"  Depth: {tree.depth if tree else 'N/A'}")
    print(f"  Nodes: {tree.num_nodes if tree else 'N/A'}")
    print(f"  Time: {our_time:.3f}s")

    # sklearn comparison
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score

        df = pd.read_csv(csv_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Encode categorical variables
        X_encoded = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])

        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y)

        # Test different sklearn configurations
        configs = [
            ("CART (default)", DecisionTreeClassifier(random_state=42)),
            ("CART (max_depth=3)", DecisionTreeClassifier(max_depth=3, random_state=42)),
            ("CART (max_depth=4)", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]

        train_size = int(0.8 * len(X_encoded))
        X_train, X_test = X_encoded[:train_size], X_encoded[train_size:]
        y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]

        print(f"\nsklearn comparison:")
        for name, clf in configs:
            timer = Timer()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            sklearn_time = timer.stop()

            sklearn_accuracy = accuracy_score(y_test, y_pred)
            print(f"  {name}: {sklearn_accuracy:.3f} (time: {sklearn_time:.3f}s)")

    except ImportError:
        print("sklearn not available for comparison")


def main():
    """Run comparisons"""
    datasets = ["data/mouse.csv", "data/car.csv"]

    for dataset in datasets:
        compare_with_sklearn(dataset)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()