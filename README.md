pip install numpy pandas scikit-learn

mac os using llvm and do the pathconfig

python setup_mac.py build_ext --inplace




python imputation_experiment.py \
    --dataset_type synthetic \
    --num_rows 10000 \
    --num_features 10 \
    --method parallel \
    --n_trees 1000 \
    --max_features 5 \
    --sample_coeff 0.2 \
    --num_threads 32 

python imputation_experiment.py \
    --dataset_type real \
    --real_data_path winequality.csv\
    --method sklearn \
    --n_trees 1000 \
    --max_features 5 \
    --sample_coeff 0.2 \
    --num_threads 32 