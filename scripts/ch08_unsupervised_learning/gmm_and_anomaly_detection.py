"""
Gaussian Mixture Models and Anomaly Detection Script.
This script demonstrates how to use GMMs for soft clustering, data generation,
and anomaly detection based on density scores. It also explores BIC/AIC for
model selection and uses Bayesian Gaussian Mixtures for automatic k-selection.
"""

import os
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def test_gmm_and_anomalies():
    data_path = "data/blobs_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Blobs data not found. Please import it first.")
        return

    print("Loading the Blobs dataset...")
    blobs_data = joblib.load(data_path)
    X = blobs_data["X"]

    # Train Gaussian Mixture Model
    print("\nTraining Gaussian Mixture Model (GMM)")
    print("GMM uses the Expectation-Maximization (EM) algorithm to fit ellipsoids.")
    # n_init=10 runs EM 10 times to avoid sub-optimal local optima.
    gm = GaussianMixture(n_components=5, n_init=10, random_state=42)
    gm.fit(X)
    print("GMM trained successfully!")

    # Soft Clustering & Generative Capabilities
    print("\nSoft Clustering & Generating New Data")
    # predict_proba returns the probability of an instance belonging to each cluster.
    probs = gm.predict_proba(X[:1]).round(3)
    print(f"Cluster probabilities for the first instance: {probs}")

    # GMM is a Generative Model. We can ask it to generate brand new data points!
    X_new, y_new = gm.sample(3)
    print(f"Generated 3 completely new instances based on learned patterns:\n{X_new.round(2)}")

    # Anomaly Detection
    print("\nAnomaly Detection using Density Scores")
    # score_samples returns the log of the probability density function (PDF).
    densities = gm.score_samples(X)

    # We define a threshold targeting the 4% of data in the lowest density regions.
    density_threshold = np.percentile(densities, 4)
    anomalies = X[densities < density_threshold]

    print(f"Density threshold (4th percentile): {density_threshold:.2f}")
    print(f"Number of anomalies detected (Outliers): {len(anomalies)} out of {len(X)}")

    # Model Selection (BIC and AIC)
    print("\nEvaluating GMM (BIC & AIC)")
    print("Inertia fails for ellipsoids. We must MINIMIZE theoretical information criteria.")
    print(f"BIC Score: {gm.bic(X):.2f}")
    print(f"AIC Score: {gm.aic(X):.2f}")

    # Bayesian Gaussian Mixture
    print("\nBayesian Gaussian Mixture (Auto k-selection)")
    print("We guess k=10, but the algorithm will zero out unnecessary clusters.")

    bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
    bgm.fit(X)

    print("\nCluster weights assigned by the Bayesian model:")
    print(bgm.weights_.round(2))
    print("Notice how it effectively eliminated 5 components by assigning them ~0.0 weight!")

    # Save Models
    print("\nSaving GMM models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(gm, os.path.join(model_dir, "gmm_model.pkl"))
    joblib.dump(bgm, os.path.join(model_dir, "bayesian_gmm_model.pkl"))
    print("Models saved successfully!")

if __name__ == "__main__":
    test_gmm_and_anomalies()
