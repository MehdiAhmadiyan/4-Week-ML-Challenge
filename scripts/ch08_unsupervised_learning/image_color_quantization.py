"""
Image Color Quantization Script.
This script demonstrates how to use K-Means clustering for image segmentation.
It treats every pixel as a data point in 3D (RGB) space, clusters them into a
smaller number of colors, and reconstructs the image using only those colors.
"""

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def quantize_image_colors():
    data_path = "data/flower_image.pkl"
    output_dir = "outputs"

    if not os.path.exists(data_path):
        print("Error: Image data not found. Please import it first.")
        return

    print("Loading the sample image...")
    image = joblib.load(data_path)

    print(f"Original image shape: {image.shape}")

    # Flatten the Image
    print("\nFlattening the Image")
    # Reshape to a 2D array: (height * width, 3 RGB channels).
    # We completely discard spatial coordinates and only keep the colors.
    X = image.reshape(-1, 3)
    print(f"Flattened dataset shape: {X.shape}")

    # Train K-Means to find top colors
    print("\nClustering Colors with K-Means")
    n_colors = 8
    print(f"Finding the top {n_colors} dominant colors...")

    # Run K-Means on the massive list of pixels
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(X)

    # Replace Colors and Reconstruct
    print("\nReconstructing the Image")
    # Replace every pixel's original color with the color of its assigned centroid.
    segmented_img_flat = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to the original image dimensions (height, width, 3).
    segmented_img = segmented_img_flat.reshape(image.shape)

    # Save and Output
    print("\nSaving the original and quantized images for comparison...")
    os.makedirs(output_dir, exist_ok=True)

    plt.imsave(os.path.join(output_dir, "original_flower.jpg"), image)
    plt.imsave(os.path.join(output_dir, f"quantized_flower_{n_colors}_colors.jpg"), segmented_img)

    print(f"Success! Check the '{output_dir}' directory to see the {n_colors}-color image.")

if __name__ == "__main__":
    quantize_image_colors()
