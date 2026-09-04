"""
PCA and Decision Boundaries Script.
This script demonstrates how Decision Trees struggle with diagonal data
(creating complex staircase boundaries) and how rotating the dataset using PCA
resolves this issue, allowing the tree to use simple straight splits.
"""

import os
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def train_rotated_tree():
    data_path = "data/iris_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Iris data not found. Please import it first.")
        return

    print("Loading the Iris dataset...")
    iris = joblib.load(data_path)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris.target

    # Create the PCA Pipeline
    print("\nRotating the Data using PCA")
    # We must scale the data before applying PCA!
    pca_pipeline = make_pipeline(StandardScaler(), PCA())

    # Transform (rotate) the original data
    X_rotated = pca_pipeline.fit_transform(X)
    print("Data successfully scaled and rotated.")

    # Train the Decision Tree on Rotated Data
    print("\nTraining Decision Tree on Rotated Data")
    tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf_pca.fit(X_rotated, y)
    print("Tree trained successfully! It can now separate the classes with simple straight lines.")

    # Save the Models (Pipeline + Tree)
    print("\nSaving the Pipeline and Model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pca_pipeline, os.path.join(model_dir, "pca_pipeline.pkl"))
    joblib.dump(tree_clf_pca, os.path.join(model_dir, "tree_clf_pca.pkl"))
    print("Models saved successfully in the 'models' directory.")
    print("Chapter 5 pipeline is COMPLETE!")

if __name__ == "__main__":
    train_rotated_tree()
