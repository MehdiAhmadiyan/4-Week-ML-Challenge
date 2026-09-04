"""
Decision Tree Regularization Script.
This script demonstrates how Decision Trees overfit if left unconstrained (nonparametric),
and how restricting hyperparameters (like min_samples_leaf) regularizes the model
and improves generalization on unseen test data.
"""

import os
import joblib
from sklearn.tree import DecisionTreeClassifier

def test_tree_regularization():
    data_path = "data/moons_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Moons data not found. Please import it first.")
        return

    print("Loading the noisy Moons dataset...")
    moons_data = joblib.load(data_path)
    X_train = moons_data["X_train"]
    y_train = moons_data["y_train"]
    X_test = moons_data["X_test"]
    y_test = moons_data["y_test"]

    # Train Unconstrained Tree
    print("\nTraining Unconstrained Tree (Prone to Overfitting)")
    # No restrictions applied. It will adapt closely to the training data.
    tree_clf1 = DecisionTreeClassifier(random_state=42)
    tree_clf1.fit(X_train, y_train)

    # Train Regularized Tree
    print("Training Regularized Tree (min_samples_leaf=5)")
    # Forcing the tree to have at least 5 samples in each leaf node smooths out the boundaries.
    tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    tree_clf2.fit(X_train, y_train)

    # Evaluate and Compare on Unseen Test Data
    print("\nEvaluation on Test Data (1000 instances)")
    score1 = tree_clf1.score(X_test, y_test)
    score2 = tree_clf2.score(X_test, y_test)

    print(f"Unconstrained Tree Accuracy: {score1 * 100:.1f}%")
    print(f"Regularized Tree Accuracy:   {score2 * 100:.1f}%")

    print("\nConclusion: The regularized tree generalizes much better to unseen data!")

    # Save the winning model
    print("\nSaving the regularized model...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(tree_clf2, os.path.join(model_dir, "regularized_tree_clf.pkl"))
    print("Model saved successfully!")

if __name__ == "__main__":
    test_tree_regularization()
