from __future__ import annotations

import os
import warnings
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import egg_info

ROOT = Path(__file__).resolve().parent
RANK_CYTHON_SOURCE = ROOT / "pi_torchreid" / "metrics" / "rank_cylib" / "rank_cy.pyx"
RANK_C_SOURCE = ROOT / "pi_torchreid" / "metrics" / "rank_cylib" / "rank_cy.c"
RANK_CYTHON_SOURCE_REL = "pi_torchreid/metrics/rank_cylib/rank_cy.pyx"
RANK_C_SOURCE_REL = "pi_torchreid/metrics/rank_cylib/rank_cy.c"
RANK_EXTENSION_NAME = "pi_torchreid.metrics.rank_cylib.rank_cy"


def env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def optional_extension_message(exc: Exception) -> str:
    return (
        "Failed to build the optional rank_cy extension; continuing without Cython acceleration. "
        "Set PI_TORCHREID_FORCE_EXT=1 to make this a hard failure or "
        "PI_TORCHREID_DISABLE_EXT=1 to skip the build attempt.\n"
        f"{exc}"
    )


def missing_extension_source_message(source_path: Path) -> str:
    return (
        "Skipping the optional rank_cy extension because the generated C source is missing: "
        f"{source_path}. "
        "Set PI_TORCHREID_USE_CYTHON=1 to build from rank_cy.pyx or "
        "PI_TORCHREID_FORCE_EXT=1 to make this a hard failure."
    )


def build_rank_extension() -> list[Extension]:
    if env_flag("PI_TORCHREID_DISABLE_EXT"):
        return []

    force_ext = env_flag("PI_TORCHREID_FORCE_EXT")
    use_cython = env_flag("PI_TORCHREID_USE_CYTHON")

    import numpy as np

    source_path = RANK_CYTHON_SOURCE if use_cython else RANK_C_SOURCE
    source_path_rel = RANK_CYTHON_SOURCE_REL if use_cython else RANK_C_SOURCE_REL
    if not source_path.exists():
        if force_ext or use_cython:
            raise FileNotFoundError(f"Missing extension source: {source_path}")

        warnings.warn(missing_extension_source_message(source_path), stacklevel=2)
        return []

    extension = Extension(
        RANK_EXTENSION_NAME,
        [source_path_rel],
        include_dirs=[np.get_include()],
    )

    if use_cython:
        try:
            from Cython.Build import cythonize
        except ImportError as exc:
            raise RuntimeError(
                "PI_TORCHREID_USE_CYTHON=1 requires Cython in the build environment."
            ) from exc

        return cythonize([extension], compiler_directives={"language_level": "3"})

    return [extension]


class OptionalBuildExt(build_ext):
    def initialize_options(self) -> None:
        super().initialize_options()
        self._optional_extension_failed = False

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:
            self._handle_optional_failure(exc)

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._handle_optional_failure(exc)

    def get_outputs(self) -> list[str]:
        return [output for output in super().get_outputs() if Path(output).exists()]

    def _handle_optional_failure(self, exc: Exception) -> None:
        if env_flag("PI_TORCHREID_FORCE_EXT"):
            raise

        if self._optional_extension_failed:
            return

        self._optional_extension_failed = True
        self.warn(optional_extension_message(exc))


class FreshEggInfo(egg_info):
    def run(self) -> None:
        sources_file = Path(self.egg_info) / "SOURCES.txt"
        if sources_file.exists():
            sources_file.unlink()

        super().run()


ext_modules = build_rank_extension()

setup(
    ext_modules=ext_modules,
    cmdclass={
        "egg_info": FreshEggInfo,
        **({"build_ext": OptionalBuildExt} if ext_modules else {}),
    },
)
