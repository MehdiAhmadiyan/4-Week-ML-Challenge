"""
Data Preparation Script for Chapter 4.
Generates linear and nonlinear (quadratic) synthetic datasets,
fetches the Iris dataset and saves them to disk for subsequent scripts.
"""

import os
import numpy as np
import joblib
from sklearn.datasets import load_iris

def prepare_and_save_data():
    data_dir = "data"

    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Set a random seed for perfect reproducibility
    rng = np.random.default_rng(seed=42)
    m = 200  # Number of instances

    # Generate linear data (y = 4 + 3x + noise)
    print("Generating linear dataset...")
    X_linear = 2 * rng.random((m, 1))
    y_linear = 4 + 3 * X_linear + rng.standard_normal((m, 1))

    linear_data = {"X": X_linear, "y": y_linear}
    joblib.dump(linear_data, os.path.join(data_dir, "linear_data.pkl"))

    # Generate nonlinear quadratic data (y = 0.5x^2 + x + 2 + noise)
    print("Generating nonlinear (quadratic) dataset...")
    X_quad = 6 * rng.random((m, 1)) - 3
    y_quad = 0.5 * X_quad ** 2 + X_quad + 2 + rng.standard_normal((m, 1))

    quad_data = {"X": X_quad, "y": y_quad}
    joblib.dump(quad_data, os.path.join(data_dir, "quad_data.pkl"))

    # Fetch the Iris dataset
    print("Fetching the Iris dataset...")
    iris = load_iris(as_frame=True)
    joblib.dump(iris, os.path.join(data_dir, "iris_data.pkl"))

    print("All datasets prepared and saved successfully in the 'data' directory!")

if __name__ == "__main__":
    prepare_and_save_data()
