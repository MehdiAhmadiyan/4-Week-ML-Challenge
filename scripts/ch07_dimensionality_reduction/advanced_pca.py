"""
Advanced PCA Techniques Script.
This script demonstrates two highly scalable PCA algorithms:
1. Randomized PCA: Uses a stochastic algorithm to quickly approximate principal components.
2. Incremental PCA (IPCA): Processes data in mini-batches to prevent Out-Of-Memory (OOM) errors.
"""

import os
import time
import joblib
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

def test_advanced_pca():
    data_path = "data/mnist_data.pkl"

    if not os.path.exists(data_path):
        print("Error: MNIST data not found. Please import it first.")
        return

    print("Loading the MNIST dataset...")
    mnist_data = joblib.load(data_path)
    X_train = mnist_data["X_train"]

    # We use 154 components because we learned from the previous script
    # that it preserves 95% of the variance for MNIST.
    n_components = 154

    # Standard PCA (For Baseline Comparison)
    print("\nStandard PCA (Baseline)")
    # svd_solver="full" forces the standard, full approach
    pca_std = PCA(n_components=n_components, svd_solver="full", random_state=42)

    t0 = time.time()
    pca_std.fit(X_train)
    t1 = time.time()
    print(f"Standard PCA Training Time:   {t1 - t0:.2f} seconds")

    # Randomized PCA
    print("\nRandomized PCA")
    print("Randomized PCA is dramatically faster when keeping a small fraction of components.")

    rnd_pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)

    t0 = time.time()
    rnd_pca.fit(X_train)
    t1 = time.time()
    print(f"Randomized PCA Training Time: {t1 - t0:.2f} seconds")

    # Incremental PCA (Mini-batch processing)
    print("\nIncremental PCA (IPCA)")
    print("IPCA splits the data into mini-batches, solving Out-Of-Memory (OOM) issues.")

    n_batches = 100
    inc_pca = IncrementalPCA(n_components=n_components)

    t0 = time.time()
    # np.array_split divides the array into 100 equal-sized mini-batches
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch) # Updates the model incrementally
    t1 = time.time()

    print(f"Incremental PCA Training Time: {t1 - t0:.2f} seconds")
    print("\nConclusion: IPCA is slightly slower than Randomized PCA due to the loop,")
    print("but it allows out-of-core learning for datasets larger than your system's RAM!")

if __name__ == "__main__":
    test_advanced_pca()
