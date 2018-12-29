from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
# setup(
#     ext_modules=cythonize("c_func.pyx"),
#     include_dirs=[numpy.get_include()]
# )
setup(ext_modules = cythonize(Extension(
    'c_func',
    sources=['c_func.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    extra_link_args=['-fopenmp']
)))