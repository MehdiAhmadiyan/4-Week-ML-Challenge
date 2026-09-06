"""
Boosting Ensembles Script.
This script demonstrates sequential ensemble learning where each new predictor
tries to correct the errors of its predecessor.
1. AdaBoostClassifier: Tweaks instance weights (using the Moons dataset).
2. GradientBoostingRegressor: Fits to residual errors with Early Stopping (using the Quadratic dataset).
"""

import os
import joblib
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, root_mean_squared_error

def train_boosting_ensembles():
    model_dir = "models"
    moons_path = "data/moons_data.pkl"
    quad_path = "data/quad_data.pkl"

    if not (os.path.exists(moons_path) and os.path.exists(quad_path)):
        print("Error: Datasets not found. Please import it first.")
        return

    print("Loading datasets...")
    moons_data = joblib.load(moons_path)
    quad_data = joblib.load(quad_path)

    # AdaBoost Classifier
    print("\nTraining AdaBoost Classifier")
    print("AdaBoost increases the relative weights of misclassified instances sequentially.")

    # We use 30 shallow trees (Decision Stumps) with max_depth=1.
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=30,
        learning_rate=0.5, random_state=42
    )

    ada_clf.fit(moons_data["X_train"], moons_data["y_train"])
    ada_pred = ada_clf.predict(moons_data["X_test"])
    ada_acc = accuracy_score(moons_data["y_test"], ada_pred)
    print(f"AdaBoost Test Accuracy: {ada_acc * 100:.1f}%")

    # Gradient Boosting Regressor (with Early Stopping)
    print("\nTraining Gradient Boosting Regressor")
    print("Gradient Boosting fits each new tree directly to the residual errors of the previous one.")

    # n_iter_no_change=10 enables Early Stopping: training halts if validation score
    # doesn't improve for 10 consecutive iterations.
    gbrt_clf = GradientBoostingRegressor(
        max_depth=2, learning_rate=0.05, n_estimators=500,
        n_iter_no_change=10, random_state=42
    )

    gbrt_clf.fit(quad_data["X"], quad_data["y"].ravel())

    # Because of Early Stopping, it shouldn't reach 500 trees.
    optimal_trees = gbrt_clf.n_estimators_
    print(f"Maximum trees allowed: 500")
    print(f"Optimal number of trees (stopped early): {optimal_trees}")

    gbrt_rmse = root_mean_squared_error(quad_data["y"], gbrt_clf.predict(quad_data["X"]))
    print(f"Gradient Boosting Training RMSE: {gbrt_rmse:.4f}")

    # Save the models
    print("\nSaving boosting models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ada_clf, os.path.join(model_dir, "adaboost_clf.pkl"))
    joblib.dump(gbrt_clf, os.path.join(model_dir, "gradient_boosting_reg.pkl"))
    print("Models saved successfully!")

if __name__ == "__main__":
    train_boosting_ensembles()
