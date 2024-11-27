#include "RandomForest.h"
#include <cmath>
#include <cstdlib>

RandomForest::RandomForest(size_t numOfTrees, size_t maxValues, size_t numLabels,
             double sampleCoeff, bool is_regression) {
    this->numOfTrees = numOfTrees;
    this->maxValues = maxValues;
    this->numLabels = numLabels;
    this->sampleCoeff = sampleCoeff;
    this->forest.resize(numOfTrees);
    this->is_regression = is_regression;
};


void RandomForest::fit(Values &X, Labels &y, const Indices &ids) {
    bootstrap = sample(ids);

#pragma omp parallel for
    for (int i = 0; i < numOfTrees; ++i) {
    const IndicesSet features = chooseFeatures(X[0].size(), maxValues);

        DecisionTree tree(is_regression);
        // train a tree with the sample
        tree.fit(X, y, bootstrap, features);
        // put it into the forest
        forest[i] = tree;
    }
}

Indices RandomForest::sample(const Indices &ids) {
    size_t data_size = ids.size();
    size_t sample_size = (int)(sampleCoeff * data_size);
    Indices idx;

    for (int i = 0; i < sample_size; ++i) {
        size_t next = rand() % data_size;  // with replacement
        idx.push_back(next);
    }
    return idx;
}


IndicesSet RandomForest::chooseFeatures(size_t numValues, size_t maxValues) {
    // randomly choose maxValues numbers from [0, numValues - 1]
    Indices idx(numValues);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    for (size_t i = 0; i < numValues; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));

    return IndicesSet(idx.begin(), idx.begin() + maxValues);
}


MutLabels RandomForest::predict(Values &X) {
    int total = X.size();
    MutLabels y(total);
// parallel
#pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        y[i] = predict(X[i]);
    }
    return y;
}

double RandomForest::predict(const Row &x) {
    if (is_regression) {
        double sum = 0.0;
        for (size_t i = 0; i < numOfTrees; ++i) {
            sum += forest[i].predict(x);
        }
        return sum / numOfTrees;  // Mean for regression
    } else {
        // Classification
        MutLabels results(numOfTrees);
        for (size_t i = 0; i < numOfTrees; ++i) {
            results[i] = forest[i].predict(x);
        }
        Counter counter(results);
        return counter.getMostFrequent();  // Mode for classification
    }
}