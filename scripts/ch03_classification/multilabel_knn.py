"""
Multilabel Classifier Training and Evaluation Script.
This script demonstrates how to output multiple binary classes for a single instance.
It trains a KNeighborsClassifier to predict if a digit is 'large' (>= 7) AND 'odd'.
"""

import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score


def train_multilabel_classifier():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"
    model_save_path = os.path.join(model_dir, "knn_multilabel_clf.pkl")

    if not os.path.exists(data_path):
        print("Error: Data not found. Please run 01_fetch_data.py first.")
        return

    print("Loading data...")
    mnist = joblib.load(data_path)
    X_train = mnist["data"][:60000]
    y_train = mnist["target"][:60000]

    # We will use the very first digit to test our model
    some_digit = X_train[0]
    actual_label = y_train[0]

    # Create the multilabel array
    print("Creating multiple labels (Is Large? Is Odd?)...")
    y_train_large = (y_train >= '7')
    y_train_odd = (y_train.astype('int8') % 2 == 1)

    # Combine them into a single array with two columns
    y_multilabel = np.c_[y_train_large, y_train_odd]

    # Train the KNN classifier
    print("Training the KNeighborsClassifier...")
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)

    # Test on a single digit
    print(f"\nPrediction for the first digit (Actual label: {actual_label}):")
    prediction = knn_clf.predict([some_digit])
    print(f"Result [Is Large?, Is Odd?]: {prediction}")

    # Evaluate the model
    print("\nEvaluating model using cross-validation...")
    print("This WILL take several minutes!")

    # Generate predictions for the whole training set
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

    # Calculate the average F1 scores
    macro_f1 = f1_score(y_multilabel, y_train_knn_pred, average="macro")
    weighted_f1 = f1_score(y_multilabel, y_train_knn_pred, average="weighted")

    print("\n--- Multilabel Evaluation ---")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(knn_clf, model_save_path)
    print(f"\nModel successfully saved at: {model_save_path}")

if __name__ == "__main__":
    train_multilabel_classifier()
