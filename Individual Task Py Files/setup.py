from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["cython_sorting.pyx", "cython_bubble_sort.pyx", "cython_quicksort.pyx"]),
    include_dirs=[numpy.get_include()]
)