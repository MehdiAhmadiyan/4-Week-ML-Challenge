"""
Data Preparation Script.
This script handles loading the data, splitting it using stratified sampling,
and building the final preprocessing pipeline (ColumnTransformer).
"""

import numpy as np
import pandas as pd
from custom_transformers import ClusterSimilarity
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)


def load_and_split_data(csv_path):
    """
    Load the dataset and split it into Train and Test sets using Stratified Sampling.
    """

    housing_full = pd.read_csv(csv_path)

    # Create income categories for stratified sampling
    housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])

    # Split the data based on income category
    strat_train_set, strat_test_set = train_test_split(
        housing_full, test_size=0.2, stratify=housing_full["income_cat"], random_state=42)

    # Drop the temporary 'income_cat' column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True) # type: ignore

    return strat_train_set, strat_test_set


# Helper functions for the ratio pipeline
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def get_preprocessing_pipeline():
    """
    Build and return the complete preprocessing pipeline (Mega-Transformer).
    """
    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler()
        )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    # Instantiate our custom transformer
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

    # Assemble all pipelines into one massive ColumnTransformer
    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=default_num_pipeline) # type: ignore

    return preprocessing
