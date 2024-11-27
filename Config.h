#ifndef __CONFIG__
#define __CONFIG__

#define JDEBUG
#define DEBUG_FOREST

#include <cstdio>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <algorithm>

using std::shared_ptr;
using std::vector;
using std::set;
using std::map;
using std::string;
using std::pair;

typedef vector<double> MutLabels;
typedef const vector<double> Labels;
typedef vector<size_t> Indices;
typedef set<size_t> IndicesSet;

#define MIN_NODE_SIZE 2

typedef vector<double> MutRow;
typedef const vector<double> Row;
typedef vector<vector<double>> MutValues;
typedef const vector<vector<double>> Values;

extern void printRow(const Row& row);

#endif
