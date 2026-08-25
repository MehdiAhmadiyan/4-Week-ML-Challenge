"""
SVM Multiclass Classifier Script.
This script demonstrates how inherently binary algorithms like SVM handle multiclass
classification. It shows Scikit-Learn's automatic One-versus-One (OvO) strategy
and how to explicitly force the One-versus-the-Rest (OvR) strategy.
Note: We use a small subset of the data because SVMs scale poorly to large datasets.
"""

import os
import joblib
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def train_svm_multiclass():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Data not found. Please run 01_fetch_data.py first.")
        return

    print("Loading data...")
    mnist = joblib.load(data_path)

    # CRITICAL: SVMs scale poorly to large datasets.
    # We only use the first 2,000 instances to save time!
    print("Extracting a small subset (2,000 instances) for SVM...")
    X_train_small = mnist["data"][:2000]
    y_train_small = mnist["target"][:2000]

    # We will use the very first digit to test our models later
    some_digit = X_train_small[0]
    actual_label = y_train_small[0]

    # Default SVM (Uses OvO Strategy Automatically)
    print("\n--- Training Default SVC (One-versus-One) ---")
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train_small, y_train_small)

    print(f"Prediction for the first digit (Actual label: {actual_label}):", svm_clf.predict([some_digit]))

    # The decision function returns 10 scores (one for each class)
    # Under the hood, it trained 45 binary classifiers and aggregated their scores!
    some_digit_scores = svm_clf.decision_function([some_digit])
    print("Decision function scores for each class (0-9):")
    print(some_digit_scores.round(2))

    # Forced OvR Strategy using OneVsRestClassifier
    print("\n--- Training Forced OvR SVM (One-versus-the-Rest) ---")
    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(X_train_small, y_train_small)

    print(f"Prediction for the first digit using OvR:", ovr_clf.predict([some_digit]))
    print(f"Number of trained models inside OvR: {len(ovr_clf.estimators_)}") # Should be 10

    # Save the default SVM model just in case we want it later
    os.makedirs(model_dir, exist_ok=True)
    svm_save_path = os.path.join(model_dir, "svm_multiclass_clf.pkl")
    print(f"\nSaving the OvO SVM model to {svm_save_path}...")
    joblib.dump(svm_clf, svm_save_path)
    print("Done!")

if __name__ == "__main__":
    train_svm_multiclass()
