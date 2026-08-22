"""
Training Script.
This script loads the prepared data, builds the full pipeline (preprocessing + model),
fine-tunes the model using RandomizedSearchCV, and saves the best model.
"""

import joblib
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Import functions from our data_prep module
from data_prep import load_and_split_data, get_preprocessing_pipeline

def main():
    # 1. Define paths (ensure these directories exist in your project structure)
    # Note: You should have a 'data' folder containing the CSV and a 'models' folder.
    csv_path = "data/housing.csv"
    model_save_path = "models/my_california_housing_model.pkl"

    print("Loading and splitting data...")
    # 2. Load the raw data and get the splits
    strat_train_set, strat_test_set = load_and_split_data(csv_path)

    # 3. Separate features and labels for the training set
    housing = strat_train_set.drop("median_house_value", axis=1) # type: ignore
    housing_labels = strat_train_set["median_house_value"].copy() # type: ignore

    # 4. Get the preprocessing Mega-Transformer
    preprocessing = get_preprocessing_pipeline()

    print("Building the full pipeline...")
    # 5. Build the full training pipeline (Preprocessing + Machine Learning Model)
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])

    # 6. Define the hyperparameter search space using statistical distributions
    param_distribs = {
        'preprocessing__geo__n_clusters': randint(low=3, high=50),
        'random_forest__max_features': randint(low=2, high=20)
    }

    print("Starting Randomized Search (this may take a while)...")
    # 7. Set up Randomized Search (10 iterations, 3-fold cross-validation)
    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42
    )

    # 8. Run the search on the raw training data
    rnd_search.fit(housing, housing_labels)

    print("Training complete! Best parameters found:")
    print(rnd_search.best_params_)

    print("Saving the best model...")
    # 9. Extract the best model and save it to disk
    final_model = rnd_search.best_estimator_
    joblib.dump(final_model, model_save_path)
    print(f"Model successfully saved at: {model_save_path}")


# This ensures the code only runs if the script is executed directly
if __name__ == "__main__":
    main()
