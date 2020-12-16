from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
	ext_modules = cythonize("./_hmmcmod.pyx", include_path=[np.get_include()])
)
