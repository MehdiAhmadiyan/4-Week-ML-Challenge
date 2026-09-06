"""
Random Forest and Feature Importance Script.
This script trains a RandomForestClassifier on the Iris dataset.
It demonstrates the algorithm's powerful ability to automatically calculate
and output the relative importance of each feature in the dataset.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def analyze_feature_importance():
    data_path = "data/iris_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Iris data not found. Please import it first.")
        return

    print("Loading the Iris dataset...")
    iris = joblib.load(data_path)
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Train the Random Forest
    print("\nTraining RandomForestClassifier (500 Trees)")
    # n_jobs=-1 enables parallel training across all CPU cores.
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(X, y)
    print("Forest trained successfully!")

    # Extract Feature Importances
    print("\nFeature Importances")
    print("Scikit-Learn calculates how much each feature reduces impurity on average.")
    print("The sum of all importances is automatically scaled to equal 1.0 (100%).\n")

    importances = rnd_clf.feature_importances_

    # Pair the feature names with their importance scores and sort them in descending order
    feature_scores = sorted(zip(importances, feature_names), reverse=True)

    for score, name in feature_scores:
        print(f"{name:<20}: {score * 100:.2f}%")

    print("\nConclusion: The model heavily relies on Petal dimensions rather than Sepal dimensions.")

    # Save the model
    print("\nSaving the Random Forest model...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rnd_clf, os.path.join(model_dir, "random_forest_clf.pkl"))
    print("Model saved successfully!")

if __name__ == "__main__":
    analyze_feature_importance()
