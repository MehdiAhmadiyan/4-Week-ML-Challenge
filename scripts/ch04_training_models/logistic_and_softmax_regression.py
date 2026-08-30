"""
Logistic and Softmax Regression Script.
This script loads the Iris dataset and trains two models:
1. A binary Logistic Regression model to detect Iris Virginica.
2. A Softmax Regression (Multinomial) model to classify all three Iris species.
It demonstrates probability estimation and decision boundaries.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_classification_models():
    data_path = "data/iris_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Iris data not found. Please import it first.")
        return

    print("Loading Iris dataset...")
    iris = joblib.load(data_path)

    # Binary Logistic Regression
    print("\nBinary Logistic Regression (Virginica Detector)")
    # We use only one feature (petal width) for simplicity
    X_bin = iris.data[["petal width (cm)"]].values
    # Target: 1 if Virginica, 0 otherwise
    y_bin = (iris.target_names[iris.target] == 'virginica').astype(int)

    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, random_state=42)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_bin, y_train_bin)

    # Predict probabilities for a flower with petal width 1.7 cm
    prob_1_7 = log_reg.predict_proba([[1.7]])
    print(f"Probability of being Virginica (width=1.7cm): {prob_1_7[0][1] * 100:.1f}%")

    # Softmax Regression (Multiclass)
    print("\nSoftmax Regression (All 3 Classes)")
    # Using two features: petal length and petal width
    X_multi = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y_multi = iris["target"] # Contains 0 (Setosa), 1 (Versicolor), 2 (Virginica)

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, random_state=42)

    # Scikit-Learn automatically uses Softmax when trained on >2 classes.
    # C is the inverse of regularization strength (higher C = less regularization).
    softmax_reg = LogisticRegression(C=30, random_state=42)
    softmax_reg.fit(X_train_multi, y_train_multi)

    # Predict a new flower (Length: 5 cm, Width: 2 cm)[cite: 4]
    new_flower = [[5, 2]]
    predicted_class = softmax_reg.predict(new_flower)[0]
    class_name = iris.target_names[predicted_class]
    probabilities = softmax_reg.predict_proba(new_flower)[0]

    print(f"New flower measurements: Length=5, Width=2")
    print(f"Predicted Class: {predicted_class} ({class_name})")
    print("Class Probabilities:")
    print(f" - Setosa:     {probabilities[0] * 100:.1f}%")
    print(f" - Versicolor: {probabilities[1] * 100:.1f}%")
    print(f" - Virginica:  {probabilities[2] * 100:.1f}%")

    # Save the models
    print("\nSaving both classification models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(log_reg, os.path.join(model_dir, "logistic_binary_model.pkl"))
    joblib.dump(softmax_reg, os.path.join(model_dir, "softmax_multi_model.pkl"))
    print("Models saved successfully. Chapter 4 pipeline is COMPLETE!")

if __name__ == "__main__":
    train_classification_models()
