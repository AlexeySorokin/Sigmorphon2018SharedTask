from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = cythonize(Extension(
    "cutility",
    sources=["cutility.pyx"],
    language="c++"
      ))
setup(ext_modules=extensions)