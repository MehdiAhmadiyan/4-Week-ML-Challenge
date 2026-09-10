"""
PCA Pipeline Tuning Script.
This script demonstrates how to treat dimensionality reduction as a hyperparameter.
It uses a Pipeline combining PCA and a RandomForestClassifier, and leverages
RandomizedSearchCV to automatically find the optimal number of PCA components
that maximizes the final classification accuracy.
"""

import os
import joblib
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV

def tune_pca_pipeline():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: MNIST data not found. Please import it first.")
        return

    print("Loading the MNIST dataset...")
    mnist_data = joblib.load(data_path)

    # We use a subset of 1000 instances to speed up the cross-validation search
    X_train_sub = mnist_data["X_train"][:1000]
    y_train_sub = mnist_data["y_train"][:1000]

    # Creating the Pipeline
    print("\nBuilding the Pipeline")
    print("Combining PCA (preprocessing) and Random Forest (classifier)...")

    pipeline = make_pipeline(
        PCA(random_state=42),
        RandomForestClassifier(random_state=42)
    )

    # Defining the Hyperparameter Distribution
    # The double underscore '__' targets the specific step in the pipeline
    param_distrib = {
        "pca__n_components": np.arange(10, 80),
        "randomforestclassifier__n_estimators": np.arange(50, 500)
    }

    # Running Randomized Search
    print("\nRunning RandomizedSearchCV")
    print("Searching for the absolute best combination of PCA dimensions and Forest trees...")

    # n_iter=10 means it will test 10 random combinations, with 3-fold cross-validation (cv=3)
    rnd_search = RandomizedSearchCV(
        pipeline, param_distrib, n_iter=10, cv=3, random_state=42, n_jobs=-1
    )

    t0 = time.time()
    rnd_search.fit(X_train_sub, y_train_sub)
    t1 = time.time()

    # Results and Saving
    print(f"\nSearch completed in {t1 - t0:.2f} seconds!")
    print("\nWinning Combination")
    print(rnd_search.best_params_)

    print(f"\nBest Cross-Validation Accuracy: {rnd_search.best_score_ * 100:.2f}%")

    print("\nSaving the optimized pipeline...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rnd_search.best_estimator_, os.path.join(model_dir, "optimized_pca_rf_pipeline.pkl"))
    print("Optimized pipeline saved successfully!")

if __name__ == "__main__":
    tune_pca_pipeline()
