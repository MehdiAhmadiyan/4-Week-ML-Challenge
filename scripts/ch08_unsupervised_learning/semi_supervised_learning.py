"""
Semi-Supervised Learning Script.
This script demonstrates Label Propagation using K-Means clustering.
Instead of labeling all instances, we cluster the data, find the most
representative instances, manually label them, and propagate these labels
to the rest of the instances to train a supervised model.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def test_label_propagation():
    # Load Data
    print("Loading the Digits dataset for Semi-Supervised Learning...")
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    k = 50  # We want to find 50 representative digits
    log_reg = LogisticRegression(max_iter=10_000, random_state=42)

    # Baseline: Train on 50 Random Instances
    print("\nBaseline: Training on 50 Random Instances")
    log_reg.fit(X_train[:k], y_train[:k])
    baseline_acc = accuracy_score(y_test, log_reg.predict(X_test))
    print(f"Baseline Accuracy: {baseline_acc * 100:.1f}%")

    # Find Representative Images
    print("\nFinding Representative Instances via Clustering")
    # By clustering first, we can find the core instances of each cluster
    kmeans = KMeans(n_clusters=k, random_state=42)

    # fit_transform returns the distance of each instance to all k centroids
    X_digits_dist = kmeans.fit_transform(X_train)

    # argmin(axis=0) finds the index of the instance closest to each centroid
    representative_digit_idx = X_digits_dist.argmin(axis=0)

    # Simulating the manual labeling of these 50 highly representative images
    y_representative_digits = y_train[representative_digit_idx]

    log_reg.fit(X_train[representative_digit_idx], y_representative_digits)
    rep_acc = accuracy_score(y_test, log_reg.predict(X_test))
    print(f"Accuracy when training on 50 Representative instances: {rep_acc * 100:.1f}%")

    # Label Propagation
    print("\nPropagating Labels to the Entire Dataset")
    # We copy the label of the representative instance to ALL other instances in its cluster
    y_train_propagated = np.empty(len(X_train), dtype=np.int32)
    for i in range(k):
        y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

    # Eliminate Outliers (The 50% Rule) ---
    print("\nEliminating Outliers for Better Accuracy")
    print("Filtering out instances near cluster boundaries to reduce noise...")

    percentile_closest = 50
    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

    for i in range(k):
        in_cluster = (kmeans.labels_ == i)
        cluster_dist = X_cluster_dist[in_cluster]

        # Calculate the distance threshold for the top 50% closest instances
        cutoff_distance = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = (X_cluster_dist > cutoff_distance)

        # Mark the outliers with -1 so we can filter them out
        X_cluster_dist[in_cluster & above_cutoff] = -1

    # Keep only the instances that are strictly NOT outliers
    partially_propagated = (X_cluster_dist != -1)
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train_propagated[partially_propagated]

    print(f"Instances kept after filtering out the furthest 50%: {len(X_train_partially_propagated)}")

    # Final Training
    print("\nFinal Results")
    log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
    final_acc = accuracy_score(y_test, log_reg.predict(X_test))
    print(f"Accuracy with Partially Propagated Labels: {final_acc * 100:.1f}%")

if __name__ == "__main__":
    test_label_propagation()
