"""
Data Preparation Script for Chapter 7 (Dimensionality Reduction).
This script fetches and generates the necessary datasets:
1. MNIST (784 dimensions) for testing PCA and data compression.
2. Swiss Roll (3 dimensions) for testing Manifold Learning algorithms like LLE.
"""

import os
import joblib
from sklearn.datasets import fetch_openml, make_swiss_roll

def prepare_and_save_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Fetching MNIST Dataset
    print("Fetching the MNIST dataset (this may take a minute or two)...")
    # Using parser='auto' to avoid warnings in newer Scikit-Learn versions
    mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

    # Splitting into train (60,000) and test (10,000)
    X_train, y_train = mnist.data[:60000], mnist.target[:60000]
    X_test, y_test = mnist.data[60000:], mnist.target[60000:]

    mnist_data = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test
    }

    mnist_path = os.path.join(data_dir, "mnist_data.pkl")
    joblib.dump(mnist_data, mnist_path)
    print(f"MNIST data saved to {mnist_path} (Features: {X_train.shape[1]})")

    # Generating Swiss Roll Dataset
    print("\nGenerating the 3D Swiss Roll dataset...")

    # -------------------------------------------------------------------------
    # Educational Note (The "Measuring Tape" Analogy):
    # Imagine a flat measuring tape. The variable 't' (or t_swiss) represents
    # the 1D position on the tape before it is rolled up (the true manifold).
    # When we roll this tape up like a Swiss roll, it exists in 3D space.
    # The variable 'X_swiss' contains the 3D coordinates (X, Y, Z) of the points.
    # We save 't' to color-code the points later, allowing us to visually
    # verify if our dimensionality reduction algorithms unrolled it correctly!
    # -------------------------------------------------------------------------
    X_swiss, t_swiss = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

    # Packaging the 3D coordinates and the 1D positions into a dictionary
    swiss_roll_data = {
        "X": X_swiss,
        "t": t_swiss
    }

    swiss_path = os.path.join(data_dir, "swiss_roll_data.pkl")
    joblib.dump(swiss_roll_data, swiss_path)
    print(f"Swiss Roll data saved to {swiss_path}")

    print("\nAll datasets prepared and saved successfully!")

if __name__ == "__main__":
    prepare_and_save_data()
