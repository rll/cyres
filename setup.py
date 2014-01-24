from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy

ceres_include = "/usr/local/include/ceres/"

ceres_lib = "/usr/local/lib/"
gflags_lib = "/usr/local/lib/"
glog_lib = "/usr/local/lib/"
cholmod_lib = amd_lib = camd_lib = colamd_lib = "/usr/local/lib/"
cxsparse_lib = "/usr/local/lib/"

ext_modules = [
    Extension(
        "cyres",
        ["cyres/src/cyres.pyx", "cyres/src/cyres.pxd", "cyres/src/ceres.pxd"],
        language="c++",
        include_dirs=[ceres_include, numpy.get_include()],
        libraries=['ceres', 'gflags', 'glog', "cholmod", "camd", "amd", "colamd", "cxsparse"],
        library_dirs=[ceres_lib, gflags_lib, glog_lib, cholmod_lib, amd_lib, camd_lib, colamd_lib, cxsparse_lib],
        extra_compile_args=['-fopenmp', '-O3'],
        extra_link_args=['-lgomp'],
    )
]

setup(
  name = 'cyres',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'cyres',
  ext_modules = ext_modules,
  packages= ['cyres'],
  package_data={'cyres': ['src/*.pxd']},
  scripts=['scripts/cyresc']
)
