"""
Data Preparation Script for Chapter 8 (Unsupervised Learning).
This script generates and fetches the necessary datasets:
1. Blobs dataset (5 clusters) for K-Means and Gaussian Mixtures.
2. Moons dataset (interleaving half-circles) for DBSCAN and Spectral Clustering.
3. A sample image for Color Quantization (Image Segmentation).
"""

import os
import joblib
import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_sample_image

def prepare_and_save_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Generate the Blobs Dataset (For K-Means/GMM)
    print("Generating the 5-Blobs dataset...")
    # Exact coordinates and standard deviations from the notebook
    blob_centers = np.array([
        [0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]
    ])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X_blobs, y_blobs = make_blobs(
        n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7
    )

    blobs_data = {"X": X_blobs, "y": y_blobs}
    joblib.dump(blobs_data, os.path.join(data_dir, "blobs_data.pkl"))
    print("Blobs dataset saved successfully.")

    # Generate the Moons Dataset (For DBSCAN)
    print("\nGenerating the Moons dataset...")
    # 1000 samples with a slight noise of 0.05 to test density-based clustering
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)

    moons_data = {"X": X_moons, "y": y_moons}
    joblib.dump(moons_data, os.path.join(data_dir, "moons_data.pkl"))
    print("Moons dataset saved successfully.")

    # Fetch a Sample Image (For Color Quantization)
    print("\nFetching a sample image for color quantization...")
    # Scikit-Learn comes with built-in images. We use 'flower.jpg' for segmentation.
    image = load_sample_image("flower.jpg")

    # It is a standard practice in ML to normalize pixel values from 0-255 to 0.0-1.0
    image = image / 255.0

    joblib.dump(image, os.path.join(data_dir, "flower_image.pkl"))
    print("Sample image saved successfully.")

    print("\nAll datasets prepared and saved successfully in the 'data' directory!")

if __name__ == "__main__":
    prepare_and_save_data()
