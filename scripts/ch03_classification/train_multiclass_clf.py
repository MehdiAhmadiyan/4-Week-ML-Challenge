"""
Multiclass Classifier Training Script.
This script trains a SGDClassifier to recognize all 10 digits (0-9).
It applies feature scaling (StandardScaler) to significantly improve performance,
and saves both the trained model and the scaler for the feature use.
"""


import os
import joblib
from numpy import cross
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def train_multiclass_classifier():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"
    model_save_path = os.path.join(model_dir, "sgd_multiclass_clf.pkl")
    scaler_save_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(data_path):
        print("Error: Data not found. Please run 01_fetch_data.py first.")
        return

    print("Loading data...")
    mnist = joblib.load(data_path)

    X_train = mnist["data"][:60000]
    y_train = mnist["target"][:60000]

    print("Scaling the features (Crucial for SGD performance)...")
    # Initialize and apply the StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

    print("Training the multiclass SGDClassifier... (This may take a few minutes)")
    # Train the model using the SCALED data and ALL 10 labels
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train_scaled, y_train)

    # Quick evaluation
    print("Evaluating with 3-fold cross-validation...")
    cv_scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Average Accuracy: {cv_scores.mean():.4f}")

    print("Saving the model and the scaler...")
    # Save both the model and the scaler to disk
    # It is important to save the scaler for the future use
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(sgd_clf, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print("Multiclass training complete!")

if __name__ == "__main__":
    train_multiclass_classifier()
