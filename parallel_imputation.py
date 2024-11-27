import numpy as np
import pandas as pd
import psutil
from parallel_imputer import RFImputer
import os
import time

def monitor_cpu_usage_during_function(func, *args, **kwargs):
    """
    Monitors and prints the CPU usage while a function is being executed.
    """
    process = psutil.Process(os.getpid())
    start_time = time.time()

    def cpu_usage_snapshot():
        return process.cpu_percent(interval=0.1) / psutil.cpu_count()

    print("Monitoring CPU usage...")
    cpu_usages = []

    result = func(*args, **kwargs)
    cpu_usages.append(cpu_usage_snapshot())

    end_time = time.time()
    print(f"CPU usage during execution: {cpu_usages}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    return result

def RandomForestImputation(df, target_features, target_dtypes, n_trees=10, max_features=2, num_labels=1, sample_coeff=0.8, num_threads = 4):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    all_features = list(df.columns)

    for target, dtype in zip(target_features, target_dtypes):
        train_mask = ~df[target].isna()
        if train_mask.sum() == 0:
            continue

        input_features = [f for f in all_features if f != target]
        X_train = df.loc[train_mask, input_features].values.astype(np.float64)
        y_train = df.loc[train_mask, target].values

        is_regression = (dtype == 'float64') or (dtype == np.float64)

        imputer = RFImputer(n_trees, max_features, num_labels, sample_coeff, is_regression)
        imputer.fit(X_train, y_train)

        X_full = df[input_features].values.astype(np.float64)
        predictions = imputer.predict(X_full)

        df[target] = df[target].fillna(pd.Series(predictions))

