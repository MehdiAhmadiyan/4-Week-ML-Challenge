"""
PCA and Data Compression Script.
This script demonstrates how to use Principal Component Analysis (PCA)
to reduce the dimensionality of the MNIST dataset while preserving 95% of its variance.
It also evaluates the compression ratio and the Reconstruction Error.
"""

import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def test_pca_compression():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: MNIST data not found. Please import it first.")
        return

    print("Loading the MNIST dataset...")
    mnist_data = joblib.load(data_path)
    X_train = mnist_data["X_train"]

    print(f"Original Training Data Shape: {X_train.shape}")
    print("Original dimensions per image: 784 (28x28 pixels)")

    # Train PCA and Compress Data
    print("\nCompressing Data with PCA")
    print("Setting n_components=0.95 to automatically preserve 95% of the variance.")

    # Initialize PCA to keep 95% of the variance
    pca = PCA(n_components=0.95, random_state=42)

    # Compress the training set
    X_reduced = pca.fit_transform(X_train)

    optimal_dimensions = pca.n_components_
    compression_ratio = (1 - (optimal_dimensions / 784)) * 100

    print(f"Dimensions required to keep 95% variance: {optimal_dimensions}")
    print(f"Compression achieved: Space reduced by {compression_ratio:.1f}%!")

    # Decompress Data and Measure Error
    print("\nDecompressing Data (Reconstruction)")
    # Projecting the reduced dataset back into the original 784D space
    X_recovered = pca.inverse_transform(X_reduced)
    print(f"Recovered Data Shape: {X_recovered.shape}")

    # Calculate the Reconstruction Error (Mean Squared Distance)
    reconstruction_error = mean_squared_error(X_train, X_recovered)
    print(f"\nReconstruction Error (MSE): {reconstruction_error:.2f}")
    print("This error represents the 5% variance (fine details/noise) that was permanently lost.")

    # Save the PCA Model
    print("\nSaving the fitted PCA model...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pca, os.path.join(model_dir, "pca_95_variance.pkl"))
    print("Model saved successfully!")

if __name__ == "__main__":
    test_pca_compression()
