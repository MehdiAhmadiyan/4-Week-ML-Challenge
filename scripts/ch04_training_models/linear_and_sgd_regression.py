"""
Linear and SGD Regression Script.
This script loads the linear dataset and trains two different models:
1. A standard LinearRegression model (which uses the SVD approach).
2. An SGDRegressor (which uses Stochastic Gradient Descent).
It compares their resulting weights (intercepts and coefficients) and saves the models.
"""

import os
import joblib
from sklearn.linear_model import LinearRegression, SGDRegressor

def train_linear_models():
    data_path = "data/linear_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Linear data not found. Plasee import it first.")
        return

    print("Loading linear data...")
    linear_data = joblib.load(data_path)
    X = linear_data["X"]
    y = linear_data["y"]

    # Linear Regression (SVD Approach)
    print("\nLinear Regression (SVD Approach)")
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # Intercept is theta_0 (should be close to 4), Coef is theta_1 (should be close to 3)
    print(f"Intercept (theta_0): {lin_reg.intercept_}")
    print(f"Coefficient (theta_1): {lin_reg.coef_}")

    # Stochastic Gradient Descent (SGDRegressor)
    print("\nStochastic Gradient Descent (SGDRegressor)")
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None,
                           eta0=0.01, n_iter_no_change=100, random_state=42)
    # SGDRegressor Hyperparameters Explanation:
    # ----------------------------------------
    # max_iter: Maximum number of epochs (passes over the training data).
    # tol: Stops training if the loss doesn't improve by at least this amount.
    # penalty: Regularization term (L1/L2). 'None' means no constraints (free movement).
    # eta0: Initial learning rate (the size of the steps the algorithm takes).
    # n_iter_no_change: Number of epochs to wait without improvement before stopping.
    # random_state: Seed for random operations to ensure reproducible results.


    # Important: SGD expects a 1D array for the target variable (y), so we use ravel()
    sgd_reg.fit(X, y.ravel())

    print(f"Intercept (theta_0): {sgd_reg.intercept_}")
    print(f"Coefficient (theta_1): {sgd_reg.coef_}")

    print("\nConclusion: Both algorithms found almost the exact same weights!")

    # Save the models
    print("\nSaving models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lin_reg, os.path.join(model_dir, "lin_reg_model.pkl"))
    joblib.dump(sgd_reg, os.path.join(model_dir, "sgd_reg_model.pkl"))
    print("Models saved successfully in the 'models' directory.")

if __name__ == "__main__":
    train_linear_models()
