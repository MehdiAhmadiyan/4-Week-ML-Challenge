"""
Random Projection Script.
This script demonstrates how to use Random Projection for dimensionality reduction.
Unlike PCA, it does not compute variance. Instead, it relies on the
Johnson-Lindenstrauss lemma to project data onto random axes while preserving distances.
"""

import os
import joblib
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection

def test_random_projection():
    data_path = "data/mnist_data.pkl"

    if not os.path.exists(data_path):
        print("Error: MNIST data not found. Please import it first.")
        return

    print("Loading a subset of the MNIST dataset...")
    mnist_data = joblib.load(data_path)

    # We use exactly 5,000 instances to match the theory from the notebook
    X_train_subset = mnist_data["X_train"][:5000]
    m = X_train_subset.shape[0]

    print(f"Dataset shape: {X_train_subset.shape} (Instances: {m})")

    # The Johnson-Lindenstrauss Lemma
    print("\nCalculating Minimum Dimensions")
    print("According to the Johnson-Lindenstrauss lemma, we calculate the minimum")
    print("dimensions (d) required to ensure a maximum distance distortion of 10%.")

    epsilon = 0.1  # 10% maximum distortion allowed
    d = johnson_lindenstrauss_min_dim(m, eps=epsilon)

    print(f"Target instances (m): {m}")
    print(f"Max allowed distortion (epsilon): {epsilon}")
    print(f"Minimum dimensions required (d):  {d}")

    print("\nNote: The required 'd' (approx 7300) is larger than our original 784 features!")
    print("This proves Random Projection is designed for ultra-high dimensional data")
    print("(e.g., text processing with millions of features), not small image datasets.")

    # Applying Sparse Random Projection
    print("\nApplying Sparse Random Projection")
    # For demonstration purposes, we will forcefully reduce it to 300 dimensions
    target_dimensions = 300
    print(f"Projecting data down to {target_dimensions} dimensions anyway...")

    # SparseRandomProjection uses a sparse matrix (mostly zeros) which makes
    # it incredibly memory-efficient and blazing fast compared to Gaussian projections.
    sparse_rnd_proj = SparseRandomProjection(n_components=target_dimensions, random_state=42)
    X_reduced = sparse_rnd_proj.fit_transform(X_train_subset)

    print(f"Reduced Data Shape: {X_reduced.shape}")
    print("Data successfully projected using completely random, sparse axes!")

if __name__ == "__main__":
    test_random_projection()
