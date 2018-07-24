from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy as np

extensions = cythonize(Extension(
           "_aligner",                                
           sources=["_aligner.pyx"],                                                   
           language="c++",
           include_dirs=[np.get_include()],
           extra_compile_args=["-std=gnu++11", "-D_hypot=hypot", "-O3"]
      ),
      annotate=True)
setup(ext_modules = extensions)