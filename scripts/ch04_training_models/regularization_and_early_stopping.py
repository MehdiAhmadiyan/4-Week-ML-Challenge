"""
Regularization and Early Stopping Script.
This script demonstrates how to constrain models to prevent overfitting using
Ridge, Lasso, and Elastic Net regularization. It also implements the
Early Stopping technique to halt training when the validation error reaches its minimum.
"""

import os
import joblib
import numpy as np
from copy import deepcopy
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

def train_regularized_models():
    data_path = "data/quad_data.pkl"

    if not os.path.exists(data_path):
        print("Error: Data not found. Please import it first.")
        return

    print("Loading data...")
    quad_data = joblib.load(data_path)
    X = quad_data["X"]
    y = quad_data["y"]

    # Split data into training and validation sets for Early Stopping
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

    print("\nRidge, Lasso, and Elastic Net")

    # Ridge (L2 penalty - Smooths the weights)
    ridge_reg = Ridge(alpha=1.0, solver="cholesky") # 'sag' solver for large datasets
    ridge_reg.fit(X_train, y_train)
    print(f"Ridge Prediction for X=1.5:       {ridge_reg.predict([[1.5]])[0][0]:.4f}")

    # Lasso (L1 penalty - Tends to push useless weights to exactly zero)
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X_train, y_train)
    print(f"Lasso Prediction for X=1.5:       {lasso_reg.predict([[1.5]])[0]:.4f}")

    # Elastic Net (Mix of L1 and L2)
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_train, y_train)
    print(f"Elastic Net Prediction for X=1.5: {elastic_net.predict([[1.5]])[0]:.4f}")

    print("\nEarly Stopping Implementation")
    # Create a highly complex pipeline (90 degrees) to easily trigger overfitting
    preprocessing = make_pipeline(
        PolynomialFeatures(degree=90, include_bias=False),
        StandardScaler()
    )

    X_train_prep = preprocessing.fit_transform(X_train)
    X_valid_prep = preprocessing.transform(X_valid)

    # SGDRegressor with penalty=None because we use Early Stopping for regularization
    sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)

    n_epochs = 500
    best_valid_rmse = float('inf')
    best_model = None
    best_epoch = 0

    # The Training Loop
    for epoch in range(n_epochs):
        # partial_fit updates the model with the current data without resetting it
        sgd_reg.partial_fit(X_train_prep, y_train.ravel())

        # Predict and evaluate on the validation set
        y_valid_predict = sgd_reg.predict(X_valid_prep)
        val_error = root_mean_squared_error(y_valid, y_valid_predict)

        # Save the model if it beats our best record
        if val_error < best_valid_rmse:
            best_valid_rmse = val_error
            best_model = deepcopy(sgd_reg) # Make a real copy of the model[cite: 4]
            best_epoch = epoch

    print(f"Early stopping simulation completed!")
    print(f"Best Validation RMSE: {best_valid_rmse:.4f} (achieved at epoch {best_epoch})")

    # Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/early_stopping_model.pkl")
    print("\nBest model saved successfully in 'models/early_stopping_model.pkl'")

if __name__ == "__main__":
    train_regularized_models()
