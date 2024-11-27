#include "DecisionTree.h"

// ids: ids to avalable data rows
// features: ids to sampled features
void DecisionTree::fit(Values &X, Labels &y,
                        const Indices &ids, const IndicesSet &features) {
    if (ids.size() == 0) {
        return; // leaves
    }

    // get the best feature to split
    double best_score = 0.0, best_value, score;
    size_t best_attr;
    Indices best_set1, best_set2;

    if (ids.size() > MIN_NODE_SIZE){
        double initial = gini(y, ids);
        // Note: if features.size() == 0, best_score = 0.0
        for (auto &attr : features) {
            // choose the threshold
            const Indices sorted_idx = argsort(X, ids, attr);
            size_t id_count = sorted_idx.size();
            double threshold = X[sorted_idx[id_count / 2]][attr];
            // divide the data set into two sets
            Indices set1, set2;
            bool missed = split(X, sorted_idx, set1, set2, attr);
            //bool missed = split(X, sorted_idx, id_count, set1, set2);

            // get the score of this attribute
            if (missed || set1.size() == 0 || set2.size() == 0) {
                score = 0.0;
            } else {
                score = gain(X, y, ids, set1, set2, initial);
            }

            // update best score
            if (score > best_score) {
                {
                    best_score = score;
                    best_attr = attr;
                    best_value = threshold;
                    best_set1 = set1;
                    best_set2 = set2;
                }

            }
        }
    }else{
        if (is_regression) {
            double sum = 0.0;
            for (size_t id : ids) {
                sum += y[id];
            }
            leaf_value = sum / ids.size();  // Store average for regression
        } else {
            leaf = std::make_shared<Counter>(y, ids);
        }

    }

    if (best_score > 0.0) {  // more attributes to test
        IndicesSet new_attr = features;
        new_attr.erase(best_attr);
        this->left = shared_ptr<DecisionTree>(new DecisionTree);
        this->right = shared_ptr<DecisionTree>(new DecisionTree);
        this->left->fit(X, y, best_set1, new_attr);
        this->right->fit(X, y, best_set2, new_attr);

        this->attr = best_attr;
        this->threshold = best_value;
        this->count = ids.size();

    } else { // all attributes tested
        if (is_regression) {
            // For regression, compute and store the mean of y[ids]
            double sum = 0.0;
            for (size_t id : ids) {
                sum += y[id];
            }
            leaf_value = sum / ids.size();
        } else {
            // For classification, create a Counter
            leaf = std::make_shared<Counter>(y, ids);
        }
        this->count = ids.size();
    }
}

MutLabels DecisionTree::predict(Values &X) {
    int total = X.size();
    MutLabels y(total);

    #pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        y[i] = predict(X[i]);
    }

    return y;
}

double DecisionTree::predict(Row &x) {
    if (leaf != nullptr) {  // leaf
        if (is_regression) {
            return leaf_value;  // Return average value for regression
        } else {
            return leaf->getMostFrequent();  // Return mode for classification
        }
    }

    double value = x[attr];
    if (value < threshold) {
        return left->predict(x);
    } else {
        return right->predict(x);
    }
}


double DecisionTree::gini(Labels &y, const Indices &ids) {
    size_t total = ids.size();
    Counter counter(y, ids);
    double imp = 0.0;
    const map<double, int> &freqs = counter.data;
    double normalized_freq;
    for (auto &freq : freqs) {
        normalized_freq = (double)freq.second / total;
        imp -= normalized_freq * normalized_freq;
    }
    return imp;
}

double DecisionTree::gain(Values &X, Labels &y, const Indices &ids,
                      const Indices &set1, const Indices &set2, double initial) {
    double p = (double)set1.size() / ids.size();
    double remainder = p * gini(y, set1) + (1 - p) * gini(y, set2);
    return initial - remainder;
}

// sort ids by values
Indices DecisionTree::argsort(Values &X, const Indices &ids, size_t attr) {
    // initialize original index locations
    Indices idx(ids.begin(), ids.end());

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&X, &attr](size_t i1, size_t i2) {
        return X[i1][attr] < X[i2][attr];
    });

    return idx;
}

bool DecisionTree::split(Values &X, const Indices &sorted_idx,
           Indices &set1, Indices &set2, size_t attr) {
    // check if out of range
    int id_count = sorted_idx.size();
    double threshold = X[sorted_idx[id_count / 2]][attr];
    if (X[sorted_idx[id_count - 1]][attr] < threshold || X[sorted_idx[0]][attr] > threshold) {
        return true;
    }

    set1 = Indices(sorted_idx.begin(), sorted_idx.begin() + id_count / 2);

    if (id_count > 1) {
        set2 = Indices(sorted_idx.begin() + id_count / 2, sorted_idx.end());
    } else {
        set2 = Indices();
    }
    return false;
}
