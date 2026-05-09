from __future__ import annotations

import numpy as np
from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "jacscanomaly._cpp_grid",
            ["src/jacscanomaly/_cpp_grid.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ]
)
