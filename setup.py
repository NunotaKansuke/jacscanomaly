from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


CPP_GRID_SOURCE = Path("src/jacscanomaly/_cpp_grid.cpp")


def _compile_and_link_args() -> tuple[list[str], list[str]]:
    """Return platform-specific flags for the mandatory C++ backend."""
    if sys.platform == "win32":
        return ["/O2", "/std:c++17", "/openmp"], []
    if sys.platform == "darwin":
        return ["-O3", "-std=c++17", "-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-O3", "-std=c++17", "-fopenmp"], ["-fopenmp"]


class build_ext(_build_ext):
    """Build the C++ backend as a required extension.

    The default PSPL workflow uses ``grid_backend='cpp'`` and
    ``single_fit_backend='cpp'``, so the package should fail during install if
    the compiled backend cannot be built instead of installing a package that
    later fails at runtime.
    """

    def build_extensions(self) -> None:
        if not CPP_GRID_SOURCE.exists():
            raise RuntimeError(
                f"Required C++ source file is missing from the source distribution: {CPP_GRID_SOURCE}."
            )

        compile_args, link_args = _compile_and_link_args()
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_args
            ext.extra_link_args = list(ext.extra_link_args or []) + link_args

        try:
            super().build_extensions()
        except Exception as exc:
            raise RuntimeError(
                "Failed to build the required jacscanomaly._cpp_grid extension. "
                "The default PSPL survey workflow needs this compiled backend. "
                "Make sure a C++17 compiler and OpenMP runtime are available, then reinstall. "
                "For a source build from PyPI, use: "
                "python -m pip install --no-binary jacscanomaly jacscanomaly"
            ) from exc


setup(
    ext_modules=[
        Extension(
            "jacscanomaly._cpp_grid",
            [str(CPP_GRID_SOURCE)],
            include_dirs=[np.get_include()],
            language="c++",
            depends=[str(CPP_GRID_SOURCE)],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
