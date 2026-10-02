"""
K-Means Clustering and Evaluation Script.
This script demonstrates Hard vs. Soft clustering and evaluates the model
using Inertia and the Silhouette Score.
"""

import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_and_evaluate_kmeans():
    data_path = "data/blobs_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Blobs data not found. Please import it first.")
        return

    print("Loading the Blobs dataset...")
    blobs_data = joblib.load(data_path)
    X = blobs_data["X"]

    # Train the K-Means Model
    print("\nTraining K-Means")
    # n_init=10 runs the algorithm 10 times with different random seeds
    # to avoid sub-optimal local minima.
    k = 5
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)

    print(f"Model trained successfully! Found {k} centroids.")

    # Hard vs. Soft Clustering
    print("\nHard vs. Soft Clustering")
    # Create some new unseen instances
    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])

    # Hard Clustering assigns the exact cluster ID.
    hard_predictions = kmeans.predict(X_new)
    print(f"Hard Clustering Predictions: {hard_predictions}")

    # Soft Clustering returns the distance to all k centroids.
    soft_predictions = kmeans.transform(X_new).round(2)
    print("Soft Clustering (Distances to all 5 centroids for the first new instance):")
    print(soft_predictions[0])

    # Model Evaluation
    print("\nEvaluation Metrics")
    # Inertia measures the sum of squared distances to the closest centroid.
    print(f"Inertia (Lower is better): {kmeans.inertia_:.2f}")

    # Silhouette score measures how well instances sit within their clusters.
    # Ranges from -1 to +1 (closer to +1 is better).
    score = silhouette_score(X, kmeans.labels_)
    print(f"Silhouette Score (-1 to 1): {score:.4f}")

    # Save the Model
    print("\nSaving the K-Means model...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_evaluate_kmeans()
