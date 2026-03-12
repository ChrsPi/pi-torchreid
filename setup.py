from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "pi_torchreid.metrics.rank_cylib.rank_cy",
        ["pi_torchreid/metrics/rank_cylib/rank_cy.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(ext_modules))
