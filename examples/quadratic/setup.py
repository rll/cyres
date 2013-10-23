from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension
import numpy
import cyres

ceres_include = "/usr/local/include/ceres/"
eigen_include = "/usr/local/include/eigen3/"

ext_modules = [
    Extension(
        "wrappers",
        ["cost_functions/wrappers.pyx"],
        language="c++",
        include_dirs=[ceres_include, numpy.get_include(), eigen_include],
        cython_include_dirs=[cyres.get_cython_include()],
        extra_objects=["cost_functions/cost_functions_impl.o"],
    )
]

setup(
  name = 'cost_functions',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'cost_functions',
  ext_modules = ext_modules,
)
