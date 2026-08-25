"""
Data Fetching Script.
This script downloads the MNIST dataset from openml and saves it locally
as a .pkl file to speed up future executions and avoid redundant downloads.
"""

import os
import joblib
from sklearn.datasets import fetch_openml

def download_and_save_mnist():
    data_dir = "data"
    save_path = os.path.join(data_dir, "mnist_data.pkl")

    # Check if the dataset has already been downloaded
    if os.path.exists(save_path):
        print(f"Dataset already exists at {save_path}. Skipping download.")
        return

    # Create the directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    print("Fetching MNIST dataset from OpenML... (This may take a minute)")
    # Download the dataset as NumPy arrays
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    print(f"Dataset fetched successfully! Saving to {save_path}...")
    joblib.dump(mnist, save_path)
    print("Done!")

if __name__ == "__main__":
    download_and_save_mnist()
