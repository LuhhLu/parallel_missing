# parallel_imputer.pyx

# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

# distutils: language = c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free

cdef extern from "RandomForest.h" namespace "":
    cdef cppclass RandomForest:
        RandomForest(size_t numOfTrees, size_t maxValues, size_t numLabels, double sampleCoeff, bool is_regression)
        void fit(vector[vector[double]] &X, vector[double] &y, vector[size_t] &ids)
        vector[double] predict(vector[vector[double]] &X)

ctypedef vector[double] vector_double
ctypedef vector[vector_double] vector_vector_double
ctypedef vector[size_t] vector_size_t

cdef class RFImputer:
    cdef RandomForest* rf

    def __cinit__(self, size_t n_trees=10, size_t max_features=5, size_t num_labels=1, double sample_coeff=0.8,
                  is_regression=False):
        self.rf = new RandomForest(n_trees, max_features, num_labels, sample_coeff, is_regression)

    def __dealloc__(self):
        if self.rf != NULL:
            del self.rf

    def fit(self, np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=1] labels):
        cdef size_t n_samples = data.shape[0]
        cdef size_t n_features = data.shape[1]

        cdef vector_vector_double X
        cdef vector_double row
        cdef vector_double y
        cdef vector_size_t ids
        cdef size_t i, j

        for i in range(n_samples):
            row.clear()
            for j in range(n_features):
                row.push_back(data[i, j])
            X.push_back(row)
            y.push_back(labels[i])
            ids.push_back(i)

        self.rf.fit(X, y, ids)

    def predict(self, np.ndarray[np.float64_t, ndim=2] data):
        cdef size_t n_samples = data.shape[0]
        cdef size_t n_features = data.shape[1]

        cdef vector_vector_double X
        cdef vector_double predictions
        cdef vector_double row
        cdef size_t i, j

        for i in range(n_samples):
            row.clear()
            for j in range(n_features):
                row.push_back(data[i, j])
            X.push_back(row)

        predictions = self.rf.predict(X)

        cdef np.ndarray[np.float64_t, ndim=1] pred_array = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            pred_array[i] = predictions[i]

        return pred_array
