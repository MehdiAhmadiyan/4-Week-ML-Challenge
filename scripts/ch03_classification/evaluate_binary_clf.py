"""
Evaluation Script for Binary Classifier.
This script loads the trained SGD model and evaluates it using more robust metrics
like Confusion Matrix, Precision, Recall, F1 Score, and ROC AUC.
"""

import os
import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_binary_classifier():
    data_path = "data/mnist_data.pkl"
    model_path = "models/sgd_binary_clf.pkl"

    # Check if required files exist
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Error: Data or model not found. Please import them")
        return

    print("Loading data and model...")
    mnist = joblib.load(data_path)
    sgd_clf = joblib.load(model_path)

    # We only need the training data for cross-validation evaluation
    X_train = mnist["data"][:60000]
    y_train = mnist["target"][:60000]
    y_train_5 = (y_train == '5')

    print("Generating cross-validated predictions... (This might take a minute)")
    # Get clean predictions for every instance in the training set
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    print("\n--- Evaluation Metrics ---")

    # Confusion Matrix
    cm = confusion_matrix(y_train_5, y_train_pred)
    print("Confusion Matrix:")
    print(cm)

    # Precision, Recall, and F1 Score
    precision = precision_score(y_train_5, y_train_pred)
    recall = recall_score(y_train_5, y_train_pred)
    f1 = f1_score(y_train_5, y_train_pred)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # ROC AUC Score (requires decision scores instead of True/False)
    print("\nCalculating decision scores for ROC AUC...")
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    roc_auc = roc_auc_score(y_train_5, y_scores)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    evaluate_binary_classifier()
