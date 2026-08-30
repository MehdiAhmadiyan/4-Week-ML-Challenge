"""
Polynomial Regression and Learning Curves Script.
This script loads the nonlinear (quadratic) dataset and demonstrates
how a simple Linear Regression underfits, while a high-degree
Polynomial Regression overfits. It visualizes these concepts using Learning Curves.
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y, title):
    """Helper function to compute and plot learning curves for a given model."""
    # Compute learning curves using 5-fold cross-validation
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
        scoring="neg_root_mean_squared_error"
    ) # type: ignore

    # Scikit-Learn returns negative RMSE scores, so we negate them to get positive errors
    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scores.mean(axis=1)

    # Plotting the results
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="Train")
    plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="Validation")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.axis([0, 160, 0, 3])
    plt.grid()
    plt.show()

def analyze_learning_curves():
    data_path = "data/quad_data.pkl"

    if not os.path.exists(data_path):
        print("Error: Quadratic data not found. Please import it first")
        return

    print("Loading nonlinear (quadratic) data...")
    quad_data = joblib.load(data_path)
    X = quad_data["X"]
    y = quad_data["y"]

    # Underfitting Example (Plain Linear Regression)
    print("\nGenerating Learning Curves for a simple Linear Model (Underfitting)...")
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y, "Underfitting: Plain Linear Regression")

    # Overfitting Example (10th-Degree Polynomial Regression)
    print("\nGenerating Learning Curves for a 10th-Degree Polynomial Model (Overfitting)...")

    # Build a pipeline: Create polynomial features -> Scale them -> Apply Linear Regression
    polynomial_regression = make_pipeline(
        PolynomialFeatures(degree=10, include_bias=False),
        StandardScaler(), # Scaling is highly recommended when creating many complex features!
        LinearRegression()
    )

    plot_learning_curves(polynomial_regression, X, y, "Overfitting: 10th-Degree Polynomial")
    print("Execution completed.")

if __name__ == "__main__":
    analyze_learning_curves()
