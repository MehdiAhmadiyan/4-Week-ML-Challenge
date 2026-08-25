"""
Random Forest Binary Classifier Script.
This script trains a RandomForestClassifier to detect the digit '5'.
It demonstrates the use of predict_proba() (instead of decision_function)
to get class probabilities and evaluates the model using F1, Precision, and Recall.
"""


import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score


def train_and_evaluate_random_forest():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"
    model_save_path = os.path.join(model_dir, "rf_binary_clf.pkl")

    if not os.path.exists(data_path):
        print("Error: Data not found. Please import it first.")
        return

    print("Loading data...")
    mnist = joblib.load(data_path)

    # We only need the training data for this evaluation
    X_train = mnist["data"][:60000]
    y_train = mnist["target"][:60000]

    # Create binary labels for the "5-detector"
    print("Preparing binary labels (True for 5s, False for others)...")
    y_train_5 = (y_train == '5')

    # Initialize the Random Forest model
    print("Initializing RandomForestClassifier...")
    forest_clf = RandomForestClassifier(random_state=42)

    # Get probabilities using cross-validation
    print("Running cross-validation to get probabilities... (This may take a minute)")
    y_probas_forest = cross_val_predict(
        forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
    )

    # Grab the probabilities for the positive class (column index 1: being a '5')
    y_scores_forest = y_probas_forest[:, 1]  # type: ignore

    # Create predictions based on the standard 50% threshold
    print("Calculating metrics using a 50% threshold...")
    y_train_pred_forest = (y_scores_forest >= 0.5)

    # Evaluate the results
    print("\n--- Random Forest Performance ---")
    print(f"F1 Score:  {f1_score(y_train_5, y_train_pred_forest):.4f}")
    print(f"Precision: {precision_score(y_train_5, y_train_pred_forest):.4f}")
    print(f"Recall:    {recall_score(y_train_5, y_train_pred_forest):.4f}")
    print("---------------------------------")

    # Finally, train the model on the full training set and save it
    print(f"\nTraining the final model and saving to {model_save_path}...")
    forest_clf.fit(X_train, y_train_5)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(forest_clf, model_save_path)

    print("Done!")

if __name__ == "__main__":
    train_and_evaluate_random_forest()
