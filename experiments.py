import pandas as pd
import numpy as np
import time
import argparse
import logging
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Import your custom parallel imputation package
from parallel_imputation import RandomForestImputation

# Configure logging
logging.basicConfig(
    filename='imputation_experiments.log',
    filemode='w',  # Overwrite the log file each run
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

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
        elif col.lower() == "target":
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
    - max_features (int): Number of features to consider when looking for the best split.
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
            logger.warning(f"Unsupported data type for column {column}. Skipping imputation.")
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

def run_experiment(experiment_id, params, df_complete, target_features, target_dtypes, log_dir):
    """
    Run a single imputation experiment using both Parallel and scikit-learn methods.

    Parameters:
    - experiment_id (str): Unique identifier for the experiment.
    - params (dict): Dictionary containing experiment parameters.
    - df_complete (pd.DataFrame): Original complete DataFrame.
    - target_features (list): List of column names to impute.
    - target_dtypes (dict): Dictionary mapping column names to dtypes.
    - log_dir (str): Directory to save individual experiment logs.

    Returns:
    - None
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Extract parameters
    num_rows = params.get('num_rows')
    num_features = params.get('num_features')
    n_trees = params.get('n_trees')
    missing_rate = params.get('missing_rate')
    num_threads = params.get('num_threads')

    # Introduce missingness
    df_missing, missing_mask = introduce_missingness(df_complete, missing_rate, random_state=42)

    # Define methods to run
    methods = ['parallel', 'sklearn']

    for method in methods:
        logger.info(f"Running Experiment {experiment_id} with method: {method}")
        logger.info(f"Parameters: Rows={num_rows}, Features={num_features}, Trees={n_trees}, Missing Rate={missing_rate*100}%, Threads={num_threads}")

        # Start timing
        start_time = time.time()

        if method == 'parallel':
            # Initialize and run the custom Parallel Random Forest Imputer
            RandomForestImputation(
                df=df_missing,
                target_features=target_features,
                target_dtypes=target_dtypes,
                n_trees=n_trees,
                max_features=int(np.sqrt(num_features)) if params.get('max_features') == 'auto' else params.get('max_features'),
                num_labels=params.get('num_labels'),
                sample_coeff=0.2,  # Adjust as per your implementation
                num_threads=num_threads
            )

            imputed_df = df_missing

        elif method == 'sklearn':
            # Run scikit-learn Random Forest Imputation
            imputed_df = sklearn_imputation(
                df=df_missing,
                target_features=target_features,
                target_dtypes=target_dtypes,
                n_trees=n_trees,
                max_features=int(np.sqrt(num_features)) if params.get('max_features') == 'auto' else params.get('max_features'),
                num_threads=num_threads
            )
        else:
            logger.error(f"Unsupported method: {method}")
            continue

        # End timing
        end_time = time.time()
        computation_time = end_time - start_time
        logger.info(f"Imputation completed in {computation_time:.2f} seconds.")

        metrics = evaluate_imputation(df_complete, imputed_df, missing_mask, target_features, target_dtypes)
        for column, metric in metrics.items():
            if 'accuracy' in metric:
                logger.info(f"Column: {column} | Accuracy: {metric['accuracy']:.4f}")
            elif 'rmse' in metric:
                logger.info(f"Column: {column} | RMSE: {metric['rmse']:.4f}")

        logger.info(f"Experiment {experiment_id} with method {method} completed.\n")

def run_real_dataset_experiment(params, log_dir):
    """
    Run imputation on a real-world dataset.

    Parameters:
    - params (dict): Dictionary containing experiment parameters.
    - log_dir (str): Directory to save the experiment log.

    Returns:
    - None
    """
    # Extract parameters
    real_data_path = params.get('real_data_path')
    method = params.get('method')
    n_trees = params.get('n_trees')
    missing_rate = params.get('missing_rate')
    max_features = params.get('max_features')
    num_threads = params.get('num_threads')

    # Load real dataset
    df_complete = pd.read_csv(real_data_path)
    target_features = [col for col in df_complete.columns if col.lower() != "target"]
    target_dtypes = determine_target_dtypes(df_complete)

    # Introduce missingness
    df_missing, missing_mask = introduce_missingness(df_complete, missing_rate, random_state=42)

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define methods to run
    methods = ['parallel', 'sklearn']

    for method in methods:
        logger.info(f"Running Real Dataset Imputation with method: {method}")
        logger.info(f"Parameters: Trees={n_trees}, Max Features={max_features}, Threads={num_threads}")

        # Start timing
        start_time = time.time()

        if method == 'parallel':
            # Initialize and run the custom Parallel Random Forest Imputer
            RandomForestImputation(
                df=df_missing,
                target_features=target_features,
                target_dtypes=target_dtypes,
                n_trees=n_trees,
                max_features=int(np.sqrt(len(target_features))) if max_features == 'auto' else max_features,
                num_labels=2,  # Adjust as per your implementation
                sample_coeff=0.2,  # Adjust as per your implementation
                num_threads=num_threads
            )
            imputed_df = df_missing

        elif method == 'sklearn':
            # Run scikit-learn Random Forest Imputation
            imputed_df = sklearn_imputation(
                df=df_missing,
                target_features=target_features,
                target_dtypes=target_dtypes,
                n_trees=n_trees,
                max_features=int(np.sqrt(len(target_features))) if max_features == 'auto' else max_features,
                num_threads=num_threads
            )
        else:
            logger.error(f"Unsupported method: {method}")
            continue

        # End timing
        end_time = time.time()
        computation_time = end_time - start_time
        logger.info(f"Imputation completed in {computation_time:.2f} seconds.")

        # Only evaluate if using synthetic data
        metrics = evaluate_imputation(df_complete, imputed_df, missing_mask, target_features, target_dtypes)
        for column, metric in metrics.items():
            if 'accuracy' in metric:
                logger.info(f"Column: {column} | Accuracy: {metric['accuracy']:.4f}")
            elif 'rmse' in metric:
                logger.info(f"Column: {column} | RMSE: {metric['rmse']:.4f}")


        logger.info(f"Real Dataset Imputation with method {method} completed.\n")

def main():
    # Define all experiments as per the provided tables

    # Table A: Varying Number of Rows and Features (Experiments 1-9)
    experiments_A = [
        {'experiment_id': '1', 'num_rows': 1000, 'num_features': 10, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '2', 'num_rows': 1000, 'num_features': 20, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '3', 'num_rows': 1000, 'num_features': 30, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '4', 'num_rows': 5000, 'num_features': 10, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '5', 'num_rows': 5000, 'num_features': 20, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '6', 'num_rows': 5000, 'num_features': 30, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '7', 'num_rows': 10000, 'num_features': 10, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '8', 'num_rows': 10000, 'num_features': 20, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '9', 'num_rows': 10000, 'num_features': 30, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 32, 'max_features': 5, 'num_labels': 2},
    ]

    # Table B: Varying Number of Trees and Threads (Experiments 10-18)
    experiments_B = [
        {'experiment_id': '10', 'num_rows': 1000, 'num_features': 30, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 4, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '11', 'num_rows': 1000, 'num_features': 30, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 8, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '12', 'num_rows': 1000, 'num_features': 30, 'n_trees': 100, 'missing_rate': 0.2, 'num_threads': 16, 'max_features': 5, 'num_labels': 2},

        {'experiment_id': '13', 'num_rows': 1000, 'num_features': 30, 'n_trees': 200, 'missing_rate': 0.2, 'num_threads': 4, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '14', 'num_rows': 1000, 'num_features': 30, 'n_trees': 200, 'missing_rate': 0.2, 'num_threads': 8, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '15', 'num_rows': 1000, 'num_features': 30, 'n_trees': 200, 'missing_rate': 0.2, 'num_threads': 16, 'max_features': 5, 'num_labels': 2},

        {'experiment_id': '16', 'num_rows': 1000, 'num_features': 30, 'n_trees': 500, 'missing_rate': 0.2, 'num_threads': 4, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '17', 'num_rows': 1000, 'num_features': 30, 'n_trees': 500, 'missing_rate': 0.2, 'num_threads': 8, 'max_features': 5, 'num_labels': 2},
        {'experiment_id': '18', 'num_rows': 1000, 'num_features': 30, 'n_trees': 500, 'missing_rate': 0.2, 'num_threads': 16, 'max_features': 5, 'num_labels': 2},
    ]

    # Table C: Real Dataset Imputation
    experiment_C = {
        'experiment_id': 'C',
        'dataset_type': 'real',
        'real_data_path': 'winequality.csv',
        'method': ['parallel', 'sklearn'],  # Both methods
        'n_trees': 100,
        'missing_rate': 0.2,
        'max_features': 'auto',
        'num_threads': 32  # Adjust as per your system
    }

    # Combine all experiments
    all_experiments = {'A': experiments_A, 'B': experiments_B, 'C': [experiment_C]}

    # Define log directory
    log_dir = 'experiment_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Run experiments A and B
    for category in ['A', 'B']:
        logger.info(f"--- Starting Experiments Category {category} ---\n")
        for exp in all_experiments[category]:
            experiment_id = exp['experiment_id']
            num_rows = exp['num_rows']
            num_features = exp['num_features']

            # Generate synthetic dataset
            df_complete = generate_large_dataset(num_rows, num_features)
            target_features = [col for col in df_complete.columns if col.lower() != "target"]
            target_dtypes = determine_target_dtypes(df_complete)

            # Run the experiment
            run_experiment(
                experiment_id=experiment_id,
                params=exp,
                df_complete=df_complete,
                target_features=target_features,
                target_dtypes=target_dtypes,
                log_dir=log_dir
            )



        logger.info(f"--- Completed Experiments Category {category} ---\n")

    # Run experiment C: Real Dataset
    logger.info(f"--- Starting Real Dataset Imputation Experiment ---\n")
    run_real_dataset_experiment(
        params=experiment_C,
        log_dir=log_dir
    )
    logger.info(f"--- Completed Real Dataset Imputation Experiment ---\n")

if __name__ == "__main__":
    main()
