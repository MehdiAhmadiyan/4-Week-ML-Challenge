"""
Bagging and Out-of-Bag (OOB) Evaluation Script.
This script trains a Bagging ensemble of 500 Decision Trees.
It demonstrates how to use the 'free' Out-of-Bag instances (the ~37% of data
not sampled by each tree) for validation, and compares the internal OOB score
with the actual test set accuracy.
"""

import os
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_bagging_ensemble():
    data_path = "data/moons_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Moons data not found. Please import it first.")
        return

    print("Loading the Moons dataset...")
    moons_data = joblib.load(data_path)
    X_train, y_train = moons_data["X_train"], moons_data["y_train"]
    X_test, y_test = moons_data["X_test"], moons_data["y_test"]

    # Train Bagging Ensemble with OOB Evaluation
    print("\nTraining BaggingClassifier (500 Trees)")

    # n_estimators: Number of trees
    # max_samples: Number of instances each tree sees
    # oob_score=True: Enables automatic evaluation on unseen instances
    # n_jobs=-1: Uses ALL available CPU cores for parallel training (Super Fast!)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, oob_score=True,
        n_jobs=-1, random_state=42
    )

    # Trees are trained independently and in parallel
    bag_clf.fit(X_train, y_train)

    # Evaluate Internal OOB Score
    print("\nEvaluation Results")
    # This is the score achieved on the 37% "unseen" training data
    print(f"Internal OOB Score (Validation estimate): {bag_clf.oob_score_ * 100:.1f}%")

    # Evaluate on Actual Test Set
    y_pred = bag_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # The OOB score is usually a very reliable estimate of the actual test score
    print(f"Actual Test Set Accuracy:                 {test_accuracy * 100:.1f}%")
    print("\nNotice how closely the OOB score estimates the final test accuracy!")

    # Save the model
    print("\nSaving the Bagging ensemble...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bag_clf, os.path.join(model_dir, "bagging_oob_clf.pkl"))
    print("Model saved successfully!")

if __name__ == "__main__":
    train_bagging_ensemble()
