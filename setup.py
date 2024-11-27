from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sysconfig

python_include = sysconfig.get_path('include')

extensions = [
    Extension(
        "parallel_imputer",
        sources=["parallel_imputer.pyx", "RandomForest.cpp", "DecisionTree.cpp", "Config.cpp"],
        include_dirs=[python_include,
                      numpy.get_include(),
                      "."],
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