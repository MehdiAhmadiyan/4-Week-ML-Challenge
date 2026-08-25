"""
Multiclass Evaluation and Error Analysis Script.
This script loads and trained multiclass model and scaler, apllies them
to the unseen TEST set and evaluates the final performance. It also visualizes
the confusion matrix using matplotlib to analyze where the model makes mistakes.
"""


import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def evaluate_multiclass_and_analyze_errors():
    data_path = "data/mnist_data.pkl"
    model_path = "models/sgd_multiclass_clf.pkl"
    scaler_path = "models/scaler.pkl"

    # Check if required files exist
    if not (os.path.exists(data_path) and os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("Error: Missing files. Please make sure all the files are imported.")
        return

    print("Loading data, model, and scaler...")
    mnist = joblib.load(data_path)
    sgd_clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Get the TEST set (the last 10,000 images)
    print("Preparing the test set...")
    X_test = mnist["data"][60000:]
    y_test = mnist["target"][60000:]

    # Scale the test data
    # CRITICAL: Use transform(), NEVER fit_transform() on test data!
    print("Scaling the test features...")
    X_test_scaled = scaler.transform(X_test.astype("float64"))

    # Make predictions on the test set
    print("Making final predictions on the test set...")
    y_test_pred = sgd_clf.predict(X_test_scaled)

    # Calculate Final Accuracy
    final_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\n=> Final Test Accuracy: {final_accuracy * 100:.2f}%\n")

    # Visualizing the Errors (Confusion Matrix)
    print("Generating Confusion Matrix plots... (Check the popup windows!)")

    # Plot 1: The standard normalized confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, normalize="true", values_format=".0%"
    )
    plt.title("Normalized Confusion Matrix (Test Set)")
    plt.show()

    # Plot 2: Focusing solely on the errors
    # Create a weight array that is True (1) for errors and False (0) for correct predictions
    sample_weight = (y_test_pred != y_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred,
        sample_weight=sample_weight,
        normalize="true", values_format=".0%"
    )
    plt.title("Errors Only (Test Set)")
    plt.show()

    print("Evaluation complete!")

if __name__ == "__main__":
    evaluate_multiclass_and_analyze_errors()
