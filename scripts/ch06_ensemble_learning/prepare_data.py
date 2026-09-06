"""
Data Preparation Script for Chapter 6 (Ensemble Learning).
This script generates and fetches the necessary datasets for this chapter:
1. The Moons dataset (for Classification Ensembles like Voting, Bagging).
2. The Iris dataset (for Random Forest Feature Importance).
3. A noisy quadratic dataset (for Gradient Boosting Regression).
All datasets are saved to the 'data' directory for modular access.
"""

import os
import joblib
import numpy as np
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split

def prepare_and_save_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Generate the Moons dataset (Classification)
    print("Generating the Moons dataset (Train and Test)...")
    # Using 1000 instances to clearly see the power of ensemble models
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.3, random_state=42)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_moons, y_moons, test_size=0.2, random_state=42
    )

    moons_data = {
        "X_train": X_m_train, "y_train": y_m_train,
        "X_test": X_m_test, "y_test": y_m_test
    }
    joblib.dump(moons_data, os.path.join(data_dir, "moons_data.pkl"))

    # Fetch the Iris dataset (Feature Importance)
    print("Fetching the Iris dataset...")
    iris = load_iris(as_frame=True)
    joblib.dump(iris, os.path.join(data_dir, "iris_data.pkl"))

    # Generate a noisy quadratic dataset (Gradient Boosting Regression)
    print("Generating a noisy quadratic dataset for Regression...")
    rng = np.random.default_rng(seed=42)
    X_quad = rng.random((200, 1)) - 0.5
    y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))

    quad_data = {"X": X_quad, "y": y_quad}
    joblib.dump(quad_data, os.path.join(data_dir, "quad_data.pkl"))

    print("\nAll datasets prepared and saved successfully in the 'data' directory!")

if __name__ == "__main__":
    prepare_and_save_data()
