"""
Data Preparation Script for Chapter 5 (Decision Trees).
This script generates and fetches all the necessary datasets for this chapter:
1. The Iris dataset (for basic classification).
2. The Moons dataset (for demonstrating regularization and overfitting).
3. A synthetic quadratic dataset (for regression trees).
It saves them all to the 'data' directory for subsequent scripts to use.
"""

import os
import joblib
import numpy as np
from sklearn.datasets import load_iris, make_moons

def prepare_and_save_data():
    data_dir = "data"

    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Fetch the Iris dataset
    print("Fetching the Iris dataset...")
    iris = load_iris(as_frame=True)
    joblib.dump(iris, os.path.join(data_dir, "iris_data.pkl"))

    # Generate the Moons dataset (Non-linear dataset for classification)
    print("Generating the Moons dataset (Train and Test)...")
    X_moons_train, y_moons_train = make_moons(n_samples=150, noise=0.2, random_state=42)
    X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)

    moons_data = {
        "X_train": X_moons_train, "y_train": y_moons_train,
        "X_test": X_moons_test, "y_test": y_moons_test
    }
    joblib.dump(moons_data, os.path.join(data_dir, "moons_data.pkl"))

    # Generate the Quadratic dataset (for Regression Trees)
    print("Generating the noisy Quadratic dataset...")
    rng = np.random.default_rng(seed=42)
    X_quad = rng.random((200, 1)) - 0.5
    y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))

    quad_data = {"X": X_quad, "y": y_quad}
    joblib.dump(quad_data, os.path.join(data_dir, "quad_data.pkl"))

    print("\nAll datasets prepared and saved successfully in the 'data' directory!")

if __name__ == "__main__":
    prepare_and_save_data()
