import os
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

def train_binary_classifier():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"
    model_save_path = os.path.join(model_dir, "sgd_binary_clf.pkl")

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}. Please run 01_fetch_data.py first.")
        return

    print("Loading the dataset from disk...")
    mnist = joblib.load(data_path)
    X, y = mnist["data"], mnist["target"]

    # Split the data (MNIST is already pre-shuffled, so we just slice it)
    print("Splitting data into train and test sets...")
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Create target vectors for the "5-detector" (True for all 5s, False for others)
    print("Creating binary labels for the '5-detector'...")
    y_train_5 = (y_train == '5')

    # Initialize and train the SGD Classifier
    print("Training the SGDClassifier... (This might take a few seconds)")
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    # Quick evaluation using Cross-Validation
    print("Evaluating the model using 3-fold cross-validation...")
    cv_scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")

    # Save the trained model
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving the trained binary model to {model_save_path}...")
    joblib.dump(sgd_clf, model_save_path)
    print("Training complete!")

if __name__ == "__main__":
    train_binary_classifier()
