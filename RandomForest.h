#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H
#include "Config.h"
#include "DecisionTree.h"

#include <fstream>
#include <sstream>
#include <cstdlib>

class RandomForest {
public:
    RandomForest(size_t numOfTrees, size_t maxValues, size_t numLabels, double sampleCoeff, bool is_regression = false);
    void fit(Values &X, Labels &y, const Indices &ids);
    MutLabels predict(Values &X);
    double predict(const Row &x);


private:
    IndicesSet chooseFeatures(size_t numValues, size_t maxValues);
    Indices sample(const Indices &ids);

    vector<DecisionTree> forest;
    MutValues X;
    MutLabels y;
    Indices ids;
    size_t numOfTrees;
    size_t maxValues;  // usually sqrt(FEATURE_NUM)
    size_t numLabels;
    double sampleCoeff;  // 1 would be good
    bool is_regression;
    Indices bootstrap;  // bootstrap sample
};

#endif