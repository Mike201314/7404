import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import random


class BinarizedData:
    """Container for binarized training data"""

    def __init__(self):
        self.data: List[List[List[bool]]] = []
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.num_features = 0

    def add_class_data(self, class_id: int, examples: List[List[bool]]):
        """Add examples for a class"""
        while len(self.data) <= class_id:
            self.data.append([])

        self.data[class_id].extend(examples)

        if examples and len(examples[0]) > self.num_features:
            self.num_features = len(examples[0])

    def set_feature_names(self, names: List[str]):
        self.feature_names = names.copy()

    def set_class_names(self, names: List[str]):
        self.class_names = names.copy()

    def cross_validation_split(self, n_folds: int, test_fold: int,
                               seed: Optional[int] = None) -> Tuple['BinarizedData', 'BinarizedData']:
        """Split for cross-validation"""
        if seed is not None:
            random.seed(seed)

        all_examples = []
        for class_id, examples in enumerate(self.data):
            for example in examples:
                all_examples.append((class_id, example.copy()))

        random.shuffle(all_examples)

        fold_size = len(all_examples) // n_folds
        test_start = test_fold * fold_size
        test_end = test_start + fold_size if test_fold < n_folds - 1 else len(all_examples)

        train_data = BinarizedData()
        test_data = BinarizedData()

        for data_obj in [train_data, test_data]:
            data_obj.feature_names = self.feature_names.copy()
            data_obj.class_names = self.class_names.copy()
            data_obj.num_features = self.num_features
            data_obj.data = [[] for _ in range(len(self.class_names))]

        for i, (class_id, example) in enumerate(all_examples):
            if test_start <= i < test_end:
                test_data.data[class_id].append(example)
            else:
                train_data.data[class_id].append(example)

        return train_data, test_data

    @property
    def num_classes(self) -> int:
        return len(self.data)


class TabularDataLoader:
    """Load tabular data from CSV"""

    def __init__(self, train_file: str, test_file: Optional[str] = None, sep: str = ','):
        self.train_file = train_file
        self.test_file = test_file
        self.sep = sep

        self.train_df = pd.read_csv(train_file, sep=sep)
        self.test_df = pd.read_csv(test_file, sep=sep) if test_file else None

        self._analyze_features()

    def _analyze_features(self):
        """Analyze feature types"""
        self.feature_types = {}
        self.feature_alphabets = {}

        feature_columns = self.train_df.columns[:-1]
        self.class_column = self.train_df.columns[-1]

        for col in feature_columns:
            try:
                pd.to_numeric(self.train_df[col].dropna())
                self.feature_types[col] = 'numeric'
                self.feature_alphabets[col] = sorted(self.train_df[col].dropna().unique())
            except (ValueError, TypeError):
                self.feature_types[col] = 'categorical'
                self.feature_alphabets[col] = sorted(self.train_df[col].unique())

        self.class_alphabet = sorted(self.train_df[self.class_column].unique())

    def get_binarized_training(self, use_equality: bool = False) -> BinarizedData:
        return self._binarize_data(self.train_df, use_equality)

    def get_binarized_testing(self, use_equality: bool = False) -> BinarizedData:
        if self.test_df is None:
            raise ValueError("No test data loaded")
        return self._binarize_data(self.test_df, use_equality)

    def _binarize_data(self, df: pd.DataFrame, use_equality: bool = False) -> BinarizedData:
        """Convert DataFrame to binary features"""
        data = BinarizedData()

        binary_features = []
        feature_names = []

        feature_columns = df.columns[:-1]

        for col in feature_columns:
            col_features, col_names = self._binarize_column(df[col], col, use_equality)
            binary_features.append(col_features)
            feature_names.extend(col_names)

        if binary_features:
            all_features = np.column_stack(binary_features)
        else:
            all_features = np.empty((len(df), 0), dtype=bool)

        class_to_id = {cls: i for i, cls in enumerate(self.class_alphabet)}

        for class_label in self.class_alphabet:
            mask = df[self.class_column] == class_label
            class_examples = all_features[mask].astype(bool).tolist()
            data.add_class_data(class_to_id[class_label], class_examples)

        data.set_feature_names(feature_names)
        data.set_class_names([str(cls) for cls in self.class_alphabet])

        return data

    def _binarize_column(self, series: pd.Series, col_name: str,
                         use_equality: bool) -> Tuple[np.ndarray, List[str]]:
        """Binarize single column"""
        alphabet = self.feature_alphabets[col_name]
        feature_type = self.feature_types[col_name]

        if len(alphabet) <= 1:
            return np.empty((len(series), 0), dtype=bool), []

        if feature_type == 'categorical' or use_equality:
            if len(alphabet) == 2:
                features = (series == alphabet[0]).values.reshape(-1, 1)
                names = [f"{col_name}=={alphabet[0]}"]
            else:
                features = np.zeros((len(series), len(alphabet)), dtype=bool)
                names = []
                for i, value in enumerate(alphabet):
                    features[:, i] = (series == value)
                    names.append(f"{col_name}=={value}")
        else:
            features = np.zeros((len(series), len(alphabet) - 1), dtype=bool)
            names = []

            for i in range(len(alphabet) - 1):
                threshold = (alphabet[i] + alphabet[i + 1]) / 2
                features[:, i] = series <= threshold
                names.append(f"{col_name}<={threshold}")

        return features, names

