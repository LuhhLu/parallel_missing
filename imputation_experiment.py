import pandas as pd
import numpy as np
import time
import argparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Import your custom parallel imputation package
from parallel_imputation import (
    RandomForestImputation
)

def determine_target_dtypes(df):
    """
    Determine the original data types of each feature based on the naming convention.
    Even-indexed features are continuous (float64), and odd-indexed features are
    categorical (int32). The 'Target' column is continuous.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns' data types are to be determined.

    Returns:
    - dict: Dictionary mapping each column name to its original data type ('float64' or 'int32').
    """
    target_dtypes = {}
    for col in df.columns:
        if col.startswith("Feature_"):
            index = int(col.split("_")[1])
            if index % 2 == 0:
                target_dtypes[col] = 'float64'
            else:
                target_dtypes[col] = 'int32'
        elif col == "Target":
            target_dtypes[col] = 'float64'
        else:
            # Default to float64 if naming convention doesn't match
            target_dtypes[col] = 'float64'
    return target_dtypes

def generate_large_dataset(num_rows, num_features):
    """
    Generate a synthetic dataset with a specified number of rows and features.
    Even-indexed features are continuous (float64), and odd-indexed features are
    categorical (int32) with unique values ≤ 10% of the number of rows.
    Missing values are NOT introduced in this function.

    Parameters:
    - num_rows (int): Number of rows in the dataset.
    - num_features (int): Number of features (columns) in the dataset.

    Returns:
    - pd.DataFrame: Generated complete DataFrame without missing values.
    """
    data = {}

    # Create features
    for i in range(num_features):
        if i % 2 == 0:
            # Even-indexed features are continuous (float64)
            data[f"Feature_{i}"] = np.random.uniform(0, 100, size=num_rows).astype(np.float64)
        else:
            # Odd-indexed features are categorical (int32) with unique values ≤ 10% of rows
            unique_values = min(10, max(2, num_rows // 10))  # Ensure at least 2 unique values
            data[f"Feature_{i}"] = np.random.choice(range(unique_values), size=num_rows).astype(np.int32)

    # Add a continuous target column
    data["Target"] = np.random.uniform(0, 50, size=num_rows).astype(np.float64)

    return pd.DataFrame(data)

def introduce_missingness(df, missing_rate, random_state=None):
    """
    Introduce missing values randomly in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - missing_rate (float): Proportion of missing values to introduce (e.g., 0.1 for 10%).
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: DataFrame with missing values introduced.
    - pd.DataFrame: Boolean DataFrame indicating where missing values were introduced.
    """
    if random_state is not None:
        np.random.seed(random_state)
    df_missing = df.copy()
    mask = np.random.rand(*df_missing.shape) < missing_rate
    df_missing = df_missing.mask(mask)
    # Convert mask to DataFrame with appropriate columns and index
    missing_mask = pd.DataFrame(mask, columns=df_missing.columns, index=df_missing.index)
    return df_missing, missing_mask


def sklearn_imputation(df, target_features, target_dtypes, n_trees, max_features, num_threads):
    """
    Perform missing value imputation using scikit-learn's Random Forest.

    Parameters:
    - df (pd.DataFrame): DataFrame with missing values.
    - target_features (list): List of column names.
    - target_dtypes (dict): Dictionary mapping column names to dtypes.
    - n_trees (int): Number of trees in the random forest.
    - max_features (str or int): Number of features to consider when looking for the best split.
    - num_threads (int): Number of threads to use.

    Returns:
    - pd.DataFrame: Imputed DataFrame.
    """
    df_imputed = df.copy()
    for column in target_features:
        if df_imputed[column].isnull().sum() == 0:
            continue  # No missing values in this column

        # Define features and target
        X = df_imputed.drop(columns=column)
        y = df_imputed[column]

        # Split into training (non-missing) and prediction (missing)
        X_train = X[y.notnull()]
        y_train = y[y.notnull()]
        X_pred = X[y.isnull()]

        if X_pred.empty:
            continue  # No missing values to impute

        # Choose model based on data type
        if target_dtypes[column] == 'int32':
            model = RandomForestClassifier(
                n_estimators=n_trees,
                max_features=max_features,
                n_jobs=num_threads,
                random_state=42
            )
        elif target_dtypes[column] == 'float64':
            model = RandomForestRegressor(
                n_estimators=n_trees,
                max_features=max_features,
                n_jobs=num_threads,
                random_state=42
            )
        else:
            print(f"Unsupported data type for column {column}. Skipping imputation.")
            continue

        # Fit the model
        model.fit(X_train, y_train)

        # Predict missing values
        y_pred = model.predict(X_pred)

        # If the target is categorical, ensure predictions are integers
        if target_dtypes[column] == 'int32':
            y_pred = y_pred.round().astype(int)

        # Fill in the missing values
        df_imputed.loc[y.isnull(), column] = y_pred

    return df_imputed


def evaluate_imputation(original_df, imputed_df, missing_mask, target_features, target_dtypes):
    """
    Evaluate the imputation by comparing imputed values to the original values.

    Parameters:
    - original_df (pd.DataFrame): Original complete DataFrame before introducing missingness.
    - imputed_df (pd.DataFrame): DataFrame after imputation.
    - missing_mask (pd.DataFrame): Boolean DataFrame indicating where missing values were introduced.
    - target_features (list): List of column names.
    - target_dtypes (dict): Dictionary mapping column names to dtypes.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    metrics = {}
    for column in target_features:
        # Identify where the missingness was introduced
        missing_col_mask = missing_mask[column]

        if not missing_col_mask.any():
            continue  # No missing values were introduced for this column

        # Extract original and imputed values where missingness was introduced
        original_values = original_df.loc[missing_col_mask, column]
        imputed_values = imputed_df.loc[missing_col_mask, column]

        # Calculate metrics based on data type
        if target_dtypes[column] == 'int32':
            # For classification, use accuracy
            acc = accuracy_score(original_values, imputed_values)
            metrics[column] = {'accuracy': acc}
        elif target_dtypes[column] == 'float64':
            # For regression, use RMSE
            rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
            metrics[column] = {'rmse': rmse}
    return metrics


def main(args):
    # Unpack arguments
    num_rows = args.num_rows
    num_features = args.num_features
    method = args.method
    n_trees = args.n_trees
    max_features = args.max_features
    num_labels = args.num_labels
    sample_coeff = args.sample_coeff
    num_threads = args.num_threads
    dataset_type = args.dataset_type
    real_data_path = args.real_data_path

    # Generate or load dataset
    if dataset_type == 'synthetic':
        print(f"Generating synthetic dataset with {num_rows} rows and {num_features} features.")
        df_complete = generate_large_dataset(num_rows, num_features)
    elif dataset_type == 'real':
        if not real_data_path:
            raise ValueError("Real data path must be provided for real dataset experiments.")
        print(f"Loading real dataset from {real_data_path}.")
        df_complete = pd.read_csv(real_data_path)
    else:
        raise ValueError("Unsupported dataset type. Choose 'synthetic' or 'real'.")

    # Determine data types
    target_features = [col for col in df_complete.columns if col != "Target"]
    target_dtypes = determine_target_dtypes(df_complete)

    if dataset_type == 'synthetic':
        print(f"Introducing missing values at a rate of {sample_coeff * 100}%.")
        df_missing, missing_mask = introduce_missingness(df_complete, sample_coeff, random_state=42)
    else:
        df_missing = df_complete.copy()
        missing_mask = pd.DataFrame(False, index=df_missing.index, columns=df_missing.columns)


    if method == 'parallel':
        print("Starting parallel Random Forest imputation using custom package.")
        start_time = time.time()

        RandomForestImputation(
            df=df_missing,
            target_features=target_features,
            target_dtypes=target_dtypes,
            n_trees=n_trees,
            max_features=max_features,
            num_labels=num_labels,
            sample_coeff=sample_coeff,
            num_threads=num_threads
        )

        imputed_df = df_missing

        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Parallel Random Forest Imputation completed in {computation_time:.2f} seconds.")

    elif method == 'sklearn':
        print("Starting scikit-learn Random Forest imputation.")
        start_time = time.time()

        imputed_df = sklearn_imputation(
            df=df_missing,
            target_features=target_features,
            target_dtypes=target_dtypes,
            n_trees=n_trees,
            max_features=max_features,
            num_threads=num_threads
        )

        end_time = time.time()
        computation_time = end_time - start_time
        print(f"scikit-learn Random Forest Imputation completed in {computation_time:.2f} seconds.")

    else:
        raise ValueError("Unsupported method. Choose 'parallel' or 'sklearn'.")

    # Evaluate imputation (only for synthetic data where original data is known)
    if dataset_type == 'synthetic':
        print("Evaluating imputation performance.")
        metrics = evaluate_imputation(df_complete, imputed_df, missing_mask, target_features, target_dtypes)
        for column, metric in metrics.items():
            if 'accuracy' in metric:
                print(f"Column: {column} | Accuracy: {metric['accuracy']:.4f}")
            elif 'rmse' in metric:
                print(f"Column: {column} | RMSE: {metric['rmse']:.4f}")
    else:
        print("Evaluation on real data is not performed as original missing values are unknown.")

    # Optionally, save the imputed dataset
    if args.save_output:
        output_path = args.output_path or f"imputed_{dataset_type}_data.csv"
        imputed_df.to_csv(output_path, index=False)
        print(f"Imputed data saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Missing Value Imputation Experiment')

    # Experiment configuration
    parser.add_argument('--num_rows', type=int, default=10000,
                        help='Number of rows in synthetic dataset (default: 10000)')
    parser.add_argument('--num_features', type=int, default=30,
                        help='Number of features in synthetic dataset (default: 30)')
    parser.add_argument('--method', type=str, choices=['parallel', 'sklearn'], default='parallel',
                        help='Imputation method to use (default: parallel)')
    parser.add_argument('--n_trees', type=int, default=100, help='Number of trees in the random forest (default: 100)')
    parser.add_argument('--max_features', type=int, default='auto',
                        help='Number of features to consider when looking for the best split (default: auto)')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels for classification (used in parallel method, default: 2)')
    parser.add_argument('--sample_coeff', type=float, default=0.2,
                        help='Proportion of missing values to introduce (e.g., 0.2 for 20%, default: 0.2)')
    parser.add_argument('--num_threads', type=int, default=-1,
                        help='Number of threads to use (-1 uses all available cores, default: -1)')

    # Dataset configuration
    parser.add_argument('--dataset_type', type=str, choices=['synthetic', 'real'], default='synthetic',
                        help='Type of dataset to use for experiments (default: synthetic)')
    parser.add_argument('--real_data_path', type=str, default='',
                        help='Path to the real-world dataset CSV file (required if dataset_type is real)')

    # Output configuration
    parser.add_argument('--save_output', action='store_true',
                        help='Whether to save the imputed dataset to a CSV file (default: False)')
    parser.add_argument('--output_path', type=str, default='',
                        help='Path to save the imputed dataset (default: imputed_<dataset_type>_data.csv)')

    args = parser.parse_args()
    main(args)
