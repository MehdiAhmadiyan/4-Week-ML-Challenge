"""
Multioutput Classifier (Noise Removal) Script.
This script demonstrates multioutput-multiclass classification.
It artificially adds noise to the MNIST dataset, trains a KNN model to map
the noisy images back to their original clean state and visualizes the result.
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def plot_digit(image_data, title="Digit"):
    """Helper function to reshape and plot a 784-pixel array."""
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.title(title)
    plt.axis("off")

def build_denoising_system():
    data_path = "data/mnist_data.pkl"
    model_dir = "models"
    model_save_path = os.path.join(model_dir, "knn_denoiser_clf.pkl")

    if not os.path.exists(data_path):
        print("Error: Data not found. Please run 01_fetch_data.py first.")
        return

    print("Loading data...")
    mnist = joblib.load(data_path)
    X_train, X_test = mnist["data"][:60000], mnist["data"][60000:]

    # Generate random noise (integers between 0 and 100)
    print("Adding artificial noise to the dataset...")
    rng = np.random.default_rng(seed=42)
    noise_train = rng.integers(0, 100, (len(X_train), 784))
    noise_test = rng.integers(0, 100, (len(X_test), 784))

    # Add noise to features
    X_train_mod = X_train + noise_train
    X_test_mod = X_test + noise_test

    # Change the targets! The target is now the ORIGINAL CLEAN IMAGE
    print("Setting targets to the original clean images...")
    y_train_mod = X_train
    y_test_mod = X_test

    # Train the model
    print("Training the KNN Denoiser... (This may take a minute)")
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)

    # Test the system on a single noisy image
    print("Testing the model on the first image of the test set...")
    noisy_digit = X_test_mod[0]
    clean_digit = knn_clf.predict([noisy_digit])

    # Visualize the transformation
    print("Opening visualization windows...")
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plot_digit(noisy_digit, title="Noisy Input")

    plt.subplot(1, 2, 2)
    plot_digit(clean_digit[0], title="Cleaned Prediction")

    plt.tight_layout()
    plt.show()

    # Save the magical denoiser model
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving the model to {model_save_path}...")
    joblib.dump(knn_clf, model_save_path)
    print("Project successfully completed!")

if __name__ == "__main__":
    build_denoising_system()
