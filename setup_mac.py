# setup_mac.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sysconfig

python_include = sysconfig.get_path('include')

llvm_prefix = os.popen('brew --prefix llvm').read().strip()

extensions = [
    Extension(
        "parallel_imputer",
        sources=["parallel_imputer.pyx", "RandomForest.cpp", "DecisionTree.cpp", "Config.cpp"],
        include_dirs=[
            numpy.get_include(),
            llvm_prefix + "/include",
            ".",
            python_include
        ],
        library_dirs=[
            llvm_prefix + "/lib"
        ],
        libraries=["omp"],
        language="c++",
        extra_compile_args=[
            "-fopenmp",
            "-O3",
            "-std=c++11"
        ],
        extra_link_args=[
            "-fopenmp"
        ],
    )
]

setup(
    name="parallel_imputer",
    ext_modules=cythonize(extensions, language_level=3),
)
