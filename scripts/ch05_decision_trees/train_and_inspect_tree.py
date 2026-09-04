"""
Train and Inspect Decision Tree Script.
This script trains a DecisionTreeClassifier on the Iris dataset,
exports the visual flowchart as a .dot file, tests class probabilities,
and uses a Depth-First Search (DFS) algorithm to inspect the low-level tree structure.
"""

import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def compute_depth(tree_clf):
    """
    Traverses the binary tree using Depth-First Search (DFS)
    to compute the depth of each node.
    """
    tree = tree_clf.tree_
    depth = np.zeros(tree.node_count)
    stack = [(0, 0)] # Stack stores tuples of (node_id, current_depth)

    while stack:
        node, node_depth = stack.pop()
        depth[node] = node_depth

        # If it's not a leaf node, add its children to the stack
        if tree.children_left[node] != tree.children_right[node]:
            stack.append((tree.children_left[node], node_depth + 1))
            stack.append((tree.children_right[node], node_depth + 1))

    return depth

def train_and_inspect_tree():
    data_path = "data/iris_data.pkl"
    output_dir = "outputs"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Iris data not found. Please import it first.")
        return

    # Create directories for outputs and models
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Loading Iris dataset...")
    iris = joblib.load(data_path)

    # We only use 2 features (petal length & width) for easy visualization
    X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y_iris = iris.target

    # Train the Model
    print("\nTraining the Decision Tree")
    # max_depth=2 prevents overfitting and keeps the tree simple
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X_iris, y_iris)
    print("Model trained successfully!")

    # Export the Visual Flowchart (.dot file)
    dot_path = os.path.join(output_dir, "my_iris_tree.dot")
    export_graphviz(
        tree_clf,
        out_file=dot_path,
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
    print(f"Visual tree flowchart saved to: {dot_path}")

    # Predictions and Probabilities
    print("\nMaking Predictions")
    new_flower = [[5, 1.5]] # Petal length=5cm, width=1.5cm
    probabilities = tree_clf.predict_proba(new_flower).round(3)
    prediction = tree_clf.predict(new_flower)

    print(f"Measurements: Length=5cm, Width=1.5cm")
    print(f"Probabilities: {probabilities[0]}")
    print(f"Predicted Class: {prediction[0]} ({iris.target_names[prediction[0]]})")

    # Low-Level Tree Inspection (Under the hood)
    print("\nInspecting Low-Level Tree Structure")
    tree = tree_clf.tree_
    print(f"Total Nodes:   {tree.node_count}")
    print(f"Max Depth:     {tree.max_depth}")
    print(f"Total Leaves:  {tree.n_leaves}")

    # Calculate depth of each node using our custom DFS algorithm
    node_depths = compute_depth(tree_clf)
    print(f"\nNode Depths (DFS Output): \n{node_depths}")

    # Check threshold values for splitting
    print(f"Splitting Thresholds (Leaf nodes show -2.0): \n{tree.threshold}")

    # Save the model
    joblib.dump(tree_clf, os.path.join(model_dir, "decision_tree_clf.pkl"))
    print("\nModel saved successfully!")

if __name__ == "__main__":
    train_and_inspect_tree()
