"""
Advanced Clustering Script.
This script explores clustering algorithms designed for non-spherical data:
1. DBSCAN: Density-based clustering (identifies core points and anomalies).
2. Spectral Clustering: Graph-based clustering (uses affinity matrices).
3. Agglomerative Clustering: Bottom-up hierarchical clustering.
"""

import os
import joblib
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

def test_advanced_clustering():
    data_path = "data/moons_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Moons data not found. Please import it first.")
        return

    print("Loading the Moons dataset (non-spherical data)...")
    moons_data = joblib.load(data_path)
    X_moons = moons_data["X"]

    # DBSCAN
    print("\nDBSCAN (Density-Based Spatial Clustering)")
    print("DBSCAN groups dense regions and marks isolated points as anomalies (-1).")

    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X_moons)

    # Check for anomalies
    anomalies_count = np.sum(dbscan.labels_ == -1)
    print(f"Total instances: {len(X_moons)}")
    print(f"Anomalies detected (label -1): {anomalies_count}")

    # The Prediction Trick: DBSCAN lacks a predict() method for new data.
    # We train a KNN classifier using only DBSCAN's Core points to predict new instances.
    print("\nTraining a KNN model on DBSCAN's Core points to enable predictions...")
    knn = KNeighborsClassifier(n_neighbors=50)
    core_samples = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    knn.fit(core_samples, core_labels)

    X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
    knn_predictions = knn.predict(X_new)
    print(f"Predictions for new instances {X_new.tolist()}:\n{knn_predictions}")

    # Spectral Clustering
    print("\nSpectral Clustering")
    print("Spectral Clustering treats data as a graph network. We use a high gamma")
    print("(short ropes) to prevent connecting the two separate moons.")

    sc = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
    sc.fit(X_moons)
    print("Spectral Clustering completed successfully on the Moons dataset.")

    # Agglomerative Clustering
    print("\nAgglomerative Clustering (Bottom-Up)")
    print("Using a simple 1D dataset to demonstrate the merging hierarchy.")

    X_simple = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
    agg = AgglomerativeClustering(linkage="complete")
    agg.fit(X_simple)

    print(f"Data points: {X_simple.ravel().tolist()}")
    print("Merge history (children_ attribute):")
    # Row 1 means point 0 and 1 merged. Row 2 means point 2 and 3 merged, etc.
    print(agg.children_)

    # Save Models
    print("\nSaving advanced clustering models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(dbscan, os.path.join(model_dir, "dbscan_model.pkl"))
    joblib.dump(sc, os.path.join(model_dir, "spectral_model.pkl"))
    joblib.dump(knn, os.path.join(model_dir, "dbscan_knn_predictor.pkl"))
    print("Models saved successfully!")

if __name__ == "__main__":
    test_advanced_clustering()
