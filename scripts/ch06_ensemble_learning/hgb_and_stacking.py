"""
HGB and Stacking Ensembles Script.
This script demonstrates two advanced ensemble techniques:
1. HistGradientBoosting (HGB): A blazing fast version of Gradient Boosting that bins data.
2. Stacking: Uses a meta-learner to combine the predictions of multiple base models.
"""

import os
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import root_mean_squared_error, accuracy_score

def train_advanced_ensembles():
    model_dir = "models"
    moons_path = "data/moons_data.pkl"
    quad_path = "data/quad_data.pkl"

    if not (os.path.exists(moons_path) and os.path.exists(quad_path)):
        print("Error: Datasets not found. Please import it first.")
        return

    print("Loading datasets...")
    moons_data = joblib.load(moons_path)
    quad_data = joblib.load(quad_path)

    # Histogram-Based Gradient Boosting (HGB)
    print("\nHistogram-Based Gradient Boosting (HGB)")
    print("HGB bins continuous features into integers (max 255), making it blazing fast.")
    print("It also natively supports missing values and categorical features.")

    # Training HGB on the quadratic dataset
    hgb_reg = HistGradientBoostingRegressor(random_state=42)
    hgb_reg.fit(quad_data["X"], quad_data["y"].ravel())

    hgb_rmse = root_mean_squared_error(quad_data["y"], hgb_reg.predict(quad_data["X"]))
    print(f"HGB Regressor Training RMSE: {hgb_rmse:.4f}")

    # Stacking (Stacked Generalization)
    print("\nStacking Classifier")
    print("Stacking uses a Meta-Learner to figure out how to best combine base models.")

    # Define base models
    base_estimators = [
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # Define the Stacking Classifier with a final Meta-Learner
    # cv=5 ensures out-of-fold predictions to prevent data leakage
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=RandomForestClassifier(random_state=43),
        cv=5
    )

    stacking_clf.fit(moons_data["X_train"], moons_data["y_train"])
    stack_pred = stacking_clf.predict(moons_data["X_test"])
    stack_acc = accuracy_score(moons_data["y_test"], stack_pred)

    print(f"Stacking Classifier Test Accuracy: {stack_acc * 100:.1f}%")

    # Save the models
    print("\nSaving advanced ensemble models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(hgb_reg, os.path.join(model_dir, "hgb_regressor.pkl"))
    joblib.dump(stacking_clf, os.path.join(model_dir, "stacking_clf.pkl"))
    print("Models saved successfully!")

if __name__ == "__main__":
    train_advanced_ensembles()
