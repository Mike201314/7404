# Optimal Decision Tree Inference

This project implements an **optimal decision tree learning algorithm** based on **SAT solving**, following the methodology of the paper:

> "Efficient Inference of Optimal Decision Trees" by Florent Avellaneda, AAAI 2020

## Features

* Supports both **depth-minimizing** and **node-minimizing** decision trees
* Implements the full **incremental learning algorithm** using SAT constraints
* Converts tabular CSV datasets into **binary features**
* Includes **cross-validation**, **tree evaluation**, and **graph visualization**


## Usage

In Terminal, run the command python demo.py

## Input Format

* CSV with the last column as the class label
* Columns may be numeric or categorical
* No missing values supported currently

Example:

## How It Works

### Step-by-step Execution:

1. **Load and Binarize Data**

   * `TabularData(file_path)` loads a CSV file
   * `get_binarized_data()` converts features to binary using one-hot or inequality encoding

2. **Cross-Validation Split**

   * The dataset is partitioned class-wise using `np.array_split`
   * For each fold, training and testing sets are created

3. **Training Decision Trees**

   * `IncrementalLearner` takes training data
   * `find_optimal_tree()` uses either:

     * Algorithm 3 (with `max_nodes`) or
     * Algorithm 2 (by increasing depth incrementally)
   * Uses `SATEncoder` to encode decision constraints into CNF
   * Solves using `pysat.Minisat22` to build optimal tree

4. **Evaluation**

   * `DecisionTree.classify()` is used for prediction
   * Metrics include accuracy, depth, node count, and elapsed time

5. **Summary Output**

   * Average metrics over all folds are printed

## Visualization

Use `TreeVisualizer` to render or export trees:


## Requirements

See `requirements.txt` 
