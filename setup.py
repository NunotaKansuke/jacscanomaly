from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


CPP_GRID_SOURCE = Path("src/jacscanomaly/_cpp_grid.cpp")
VBM_CPP_SOURCE = Path("src/jacscanomaly/_vbm_cpp.cpp")
PARALLAX_CPP_SOURCE = Path("src/jacscanomaly/_parallax_cpp.cpp")


def _find_vbm_sources() -> tuple[Path, Path] | None:
    """Locate the C++ sources installed by the VBMicrolensing build dependency.

    Native FSPL/parallax support is part of the standard build.
    ``VBMICROLENSING_SOURCE_DIR`` supports non-standard installations.
    """
    candidates: list[Path] = []
    source_dir = os.environ.get("VBMICROLENSING_SOURCE_DIR")
    if source_dir:
        candidates.append(Path(source_dir))
    try:
        import VBMicrolensing  # type: ignore

        candidates.append(Path(VBMicrolensing.__file__).resolve().parent / "lib")
    except ImportError:
        pass
    for directory in candidates:
        if (directory / "VBMicrolensingLibrary.cpp").is_file() and (directory / "VBMicrolensingLibrary.h").is_file():
            return directory / "VBMicrolensingLibrary.cpp", directory
    return None


def _compile_and_link_args() -> tuple[list[str], list[str]]:
    """Return platform-specific flags for the mandatory C++ backend."""
    if sys.platform == "win32":
        return ["/O2", "/std:c++17", "/openmp"], []
    if sys.platform == "darwin":
        # Apple does not ship an OpenMP runtime.  macOS wheels disable the
        # OpenMP pragmas so they do not depend on the Homebrew libomp version
        # or on a particular macOS deployment target.  Local source builds
        # keep OpenMP enabled unless this switch is set explicitly.
        if os.environ.get("JACSCANOMALY_DISABLE_OPENMP", "").lower() in {"1", "true", "yes"}:
            return ["-O3", "-std=c++17"], []
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

        vbm_sources = _find_vbm_sources()
        if vbm_sources is None:
            raise RuntimeError(
                "VBMicrolensing C++ sources are required to build jacscanomaly's "
                "native FSPL/parallax backend. Install VBMicrolensing>=5.5 or set "
                "VBMICROLENSING_SOURCE_DIR."
            )
        vbm_library, vbm_include = vbm_sources
        compile_args, link_args = _compile_and_link_args()
        for ext in self.extensions:
            if ext.name in {"jacscanomaly._vbm_cpp", "jacscanomaly._parallax_cpp"}:
                # Keep the external VBM source out of ``Extension.sources``
                # until wheel compilation.  ``sdist`` otherwise tries to copy
                # a path outside this repository.  The isolated build
                # environment installs VBMicrolensing from pyproject.toml.
                ext.sources = list(ext.sources or []) + [str(vbm_library)]
                ext.include_dirs = list(ext.include_dirs or []) + [str(vbm_include)]
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


extensions = [
        Extension(
            "jacscanomaly._cpp_grid",
            [str(CPP_GRID_SOURCE)],
            include_dirs=[np.get_include()],
            language="c++",
            depends=[str(CPP_GRID_SOURCE)],
        )
]

extensions.append(
    Extension(
        "jacscanomaly._vbm_cpp",
        [str(VBM_CPP_SOURCE)],
        include_dirs=[np.get_include()],
        language="c++",
        depends=[str(VBM_CPP_SOURCE)],
    )
)
extensions.append(
    Extension(
        "jacscanomaly._parallax_cpp",
        [str(PARALLAX_CPP_SOURCE)],
        include_dirs=[np.get_include(), str(PARALLAX_CPP_SOURCE.parent)],
        language="c++",
        depends=[
            str(PARALLAX_CPP_SOURCE),
            "src/jacscanomaly/cpp/ephemeris.hpp",
            "src/jacscanomaly/cpp/sky_projection.hpp",
            "src/jacscanomaly/cpp/parallax_trajectory.hpp",
            "src/jacscanomaly/cpp/vbm_magnification.hpp",
        ],
    )
)

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
)
