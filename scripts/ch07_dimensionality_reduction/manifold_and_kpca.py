"""
Manifold Learning and Kernel PCA Script.
This script demonstrates how to unroll nonlinear datasets (like the Swiss Roll)
using Locally Linear Embedding (LLE) and Kernel PCA (kPCA).
Standard PCA fails here because it squashes the manifold, mixing the layers.
"""

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA

def test_manifold_learning():
    data_path = "data/swiss_roll_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Swiss Roll data not found. Please import it first.")
        return

    print("Loading the 3D Swiss Roll dataset...")
    swiss_data = joblib.load(data_path)
    X = swiss_data["X"]
    t = swiss_data["t"]  # The 1D positions used for color-coding the points

    # Locally Linear Embedding (LLE)
    print("\nUnrolling with LLE")
    print("LLE preserves local distances (k-nearest neighbors) to unroll the manifold.")

    # n_neighbors=10 means it evaluates linear relationships locally for each point
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_unrolled_lle = lle.fit_transform(X)

    print("LLE transformation complete!")

    # Kernel PCA (kPCA)
    print("\nUnrolling with Kernel PCA (RBF Kernel)")
    print("kPCA uses the kernel trick to perform complex, non-linear projections.")

    # Using the RBF kernel to separate the twisted layers
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.002, random_state=42)
    X_unrolled_kpca = rbf_pca.fit_transform(X)

    print("Kernel PCA transformation complete!")

    # Visualizing the Results
    print("\nVisualizing the Unrolled Manifolds")
    print("A plot window should open. Close it to continue and save the models.")

    plt.figure(figsize=(12, 5))

    # Plot 1: LLE
    plt.subplot(1, 2, 1)
    # Using 'c=t' applies our "measuring tape" trick to color-code the unrolled points
    plt.scatter(X_unrolled_lle[:, 0], X_unrolled_lle[:, 1], c=t, cmap=plt.cm.hot)
    plt.title("Unrolled using LLE")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Plot 2: Kernel PCA (RBF)
    plt.subplot(1, 2, 2)
    plt.scatter(X_unrolled_kpca[:, 0], X_unrolled_kpca[:, 1], c=t, cmap=plt.cm.hot)
    plt.title("Unrolled using kPCA (RBF)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()
    plt.show() # This will pause execution until you close the matplotlib window

    # Save the Models
    print("\nSaving Manifold Learning models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lle, os.path.join(model_dir, "lle_model.pkl"))
    joblib.dump(rbf_pca, os.path.join(model_dir, "kpca_rbf_model.pkl"))
    print("Models saved successfully!")

if __name__ == "__main__":
    test_manifold_learning()
