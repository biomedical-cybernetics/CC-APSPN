"""
Setup script for fast_apspn_kernel package
"""

from pathlib import Path
import sys

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
readme_path = REPO_ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else "Python wrappers for CC-APSPN."
LIB_SUFFIX = ".dll" if sys.platform == "win32" else ".so"
LIBRARIES = ("brandes_apspn_kernel", "cm_gc", "cc_sebccopts_apspn_kernel",
             "cc_apspn_kernel_approx", "cc_apspn_kernel_par")


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class NativeWheel(bdist_wheel):
    def get_tag(self):
        # ctypes uses the C ABI, not a Python-version-specific extension ABI.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


class CheckedBuild(build_py):
    def run(self):
        missing = [name + LIB_SUFFIX for name in LIBRARIES
                   if not (PACKAGE_DIR / "fast_apspn_kernel" / "lib" / (name + LIB_SUFFIX)).is_file()]
        if missing:
            raise RuntimeError("Build native libraries first with source/python/build_native.py; missing: " + ", ".join(missing))
        super().run()
        # Keep the repository's license texts with the distributed native libraries.
        license_dir = Path(self.build_lib) / "fast_apspn_kernel" / "licenses"
        self.mkpath(str(license_dir))
        for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            self.copy_file(str(REPO_ROOT / filename), str(license_dir / filename))

setup(
    name="fast_apspn_kernel",
    version="1.0.0",
    author="Junda Chen",
    author_email="kalokagathos.agon@gmail.com",
    description="Python wrappers for high-performance APSPN-based algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biomedical-cybernetics/CC-APSPN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov"],
    },
    include_package_data=False,
    package_data={
        "fast_apspn_kernel": ["lib/*" + LIB_SUFFIX],
    },
    distclass=NativeDistribution,
    cmdclass={"bdist_wheel": NativeWheel, "build_py": CheckedBuild},
)

