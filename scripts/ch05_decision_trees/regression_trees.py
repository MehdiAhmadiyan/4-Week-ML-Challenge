"""
Regression Trees Script.
This script trains DecisionTreeRegressors on a noisy quadratic dataset.
It demonstrates how unconstrained regression trees severely overfit the data
and how applying min_samples_leaf=10 regularizes the predictions,
creating a smoother step function.
"""

import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

def train_regression_trees():
    data_path = "data/quad_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Quadratic data not found. Please import it first.")
        return

    print("Loading the noisy quadratic dataset...")
    quad_data = joblib.load(data_path)
    X = quad_data["X"]
    y = quad_data["y"]

    # Train Unconstrained Regression Tree
    print("\nTraining Unconstrained Regression Tree")
    # No restrictions. The tree will memorize the noise.
    tree_reg1 = DecisionTreeRegressor(random_state=42)
    tree_reg1.fit(X, y)

    rmse1 = root_mean_squared_error(y, tree_reg1.predict(X))
    print(f"Unconstrained Tree RMSE (on training data): {rmse1:.4f}")
    print("(A score very close to 0 indicates severe overfitting to the noise!)")

    # Train Regularized Regression Tree
    print("\nTraining Regularized Regression Tree")
    # min_samples_leaf=10 forces the tree to average at least 10 instances per leaf.
    tree_reg2 = DecisionTreeRegressor(min_samples_leaf=10, random_state=42)
    tree_reg2.fit(X, y)

    rmse2 = root_mean_squared_error(y, tree_reg2.predict(X))
    print(f"Regularized Tree RMSE (on training data):   {rmse2:.4f}")
    print("(Higher training error, but a much smoother and logical generalization.)")

    # Demonstrating the 'Step Function' Behavior
    print("\nStep Function Demonstration")
    print("Because regression trees output the average value of a leaf node")
    print("nearby points falling in the same leaf get the EXACT SAME prediction.")

    # Test a few points very close to each other on the X axis
    X_test_points = np.array([[0.200], [0.201], [0.202]])
    preds1 = tree_reg1.predict(X_test_points).round(4)
    preds2 = tree_reg2.predict(X_test_points).round(4)

    print("\nPredictions for X = [0.200, 0.201, 0.202]:")
    print(f"Unconstrained Tree: {preds1} (Reacts wildly to tiny noise variations)")
    print(f"Regularized Tree:   {preds2} (Perfectly flat step!)")

    # Save Models
    print("\nSaving regression models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(tree_reg1, os.path.join(model_dir, "unconstrained_tree_reg.pkl"))
    joblib.dump(tree_reg2, os.path.join(model_dir, "regularized_tree_reg.pkl"))
    print("Models saved successfully!")

if __name__ == "__main__":
    train_regression_trees()
