"""
Evaluation Script.
This script loads the pre-trained model and evaluates it on the unseen test set,
reporting the final RMSE and 95% confidence interval.
"""

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import root_mean_squared_error

# Import our custom function to get the test data
from data_prep import load_and_split_data
# CRITICAL: We must import custom transformers so joblib can successfully unpickle the model
from custom_transformers import ClusterSimilarity, StandardScalerClone

def main():
    csv_path = "data/housing.csv"
    model_load_path = "models/my_california_housing_model.pkl"

    print("Loading test data...")
    # We only need the test set this time, so we ignore the train set using '_'
    _, strat_test_set = load_and_split_data(csv_path)

    # Separate features and labels
    X_test = strat_test_set.drop("median_house_value", axis=1)  # type: ignore
    y_test = strat_test_set["median_house_value"].copy()  # type: ignore

    print("Loading the trained model...")
    # Load the model from disk with error handling
    try:
        final_model = joblib.load(model_load_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_load_path}.")
        print("Please run train.py first to train and save the model.")
        return

    print("Evaluating the model on the test set...")
    # Make predictions (NEVER use fit() or fit_transform() on the test set)
    final_predictions = final_model.predict(X_test)

    # Calculate RMSE
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    print(f"\n=> Final Test RMSE: ${final_rmse:,.2f}")

    # Compute a 95% confidence interval for the test RMSE
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    boot_result = stats.bootstrap([squared_errors], np.mean, confidence_level=confidence, random_state=42)
    rmse_lower = np.sqrt(boot_result.confidence_interval.low)
    rmse_upper = np.sqrt(boot_result.confidence_interval.high)

    print(f"=> 95% Confidence Interval for RMSE: (${rmse_lower:,.2f}, ${rmse_upper:,.2f})")
    print("\nProject successfully completed! The model is ready for production.")

if __name__ == "__main__":
    main()
