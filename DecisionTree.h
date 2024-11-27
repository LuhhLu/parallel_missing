#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "Config.h"

#include <chrono>
#include <random>
class Counter {
public:
    Counter(Labels &y, const Indices &ids) {
        for (auto &id : ids) {
            if (data.find(y[id]) != data.end()) {
                data[y[id]] += 1;
            } else {
                data[y[id]] = 1;
            }
        }
    }

    Counter(Labels &y) {
        for (auto &label : y) {
            if (data.find(label) != data.end()) {
                data[label] += 1;
            } else {
                data[label] = 1;
            }
        }
    }

    double getMostFrequent() const {  // 返回 double
        std::vector<pair<double, int>> pairs(data.begin(), data.end());
        std::sort(pairs.begin(), pairs.end(), [=](const pair<double, int>& a, const pair<double, int>& b) {
                return a.second > b.second;
            }
        );
        return pairs.begin()->first;
    }

    map<double, int> data;  // 修改为 map<double, int>
};

class DecisionTree {

public:
    DecisionTree(bool is_regression = false) : is_regression(is_regression), left(nullptr), right(nullptr), leaf(nullptr) {}
    // rules of three

    DecisionTree(const DecisionTree &other) {
        this->left = other.left;
        this->right = other.right;
        this->leaf = other.leaf;
        this->attr = other.attr;
        this->threshold = other.threshold;
        this->count = other.count;
    }

    void swap(DecisionTree &other) {
        std::swap(this->left, other.left);
        std::swap(this->right, other.right);
        std::swap(this->leaf, other.leaf);
        std::swap(this->attr, other.attr);
        std::swap(this->threshold, other.threshold);
        std::swap(this->count, other.count);
    }

    DecisionTree &operator=(const DecisionTree &other) {
        DecisionTree temp(other);
        temp.swap(*this);
        return *this;
    }

    ~DecisionTree() {
        // all automatically recycled
    }

    void print(int indent = 2);

    void fit(Values &X, Labels &y, const Indices &ids,
             const IndicesSet &features);

    MutLabels predict(Values &X);
    double predict(Row &x);
private:
    double gini(Labels &y, const Indices &ids);
    double gain(Values &X, Labels &y, const Indices &ids,
                const Indices &set1, const Indices &set2, double initial);
    Indices argsort(Values &X, const Indices &ids, size_t attr);
    bool split(Values &X, const Indices &sorted_idx,
               Indices &set1, Indices &set2, size_t attr);

    std::shared_ptr<DecisionTree> left;
    std::shared_ptr<DecisionTree> right;
    double leaf_value; // For regression
    bool is_regression; // Task type

    size_t attr;
    double threshold;
    size_t count;

    std::shared_ptr<Counter> leaf;
};

#endif
