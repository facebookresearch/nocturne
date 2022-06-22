# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run via ```python setup.py develop``` to install Nocturne in your environment."""
import logging
import multiprocessing
import os
import re
import subprocess
import sys

from distutils.version import LooseVersion
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Reference:
# https://www.benjack.io/2017/06/12/python-cpp-tests.html


class CMakeExtension(Extension):
    """Use CMake to construct the Nocturne extension."""

    def __init__(self, name, src_dir=""):
        Extension.__init__(self, name, sources=[])
        self.src_dir = os.path.abspath(src_dir)


class CMakeBuild(build_ext):
    """Utility class for building Nocturne."""

    def run(self):
        """Run cmake."""
        try:
            cmake_version = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", cmake_version.decode()).group(1))
        if cmake_version < "3.14":
            raise RuntimeError("CMake >= 3.14 is required.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """Run the C++ build commands."""
        ext_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + ext_dir,
            "-DPYTHON_EXECUTABLE=" + sys.executable
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--", f"-j{multiprocessing.cpu_count()}"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} \
                -DVERSION_INFO="{self.distribution.get_version()}"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmd = ["cmake", ext.src_dir] + cmake_args
        try:
            subprocess.check_call(cmd, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError:
            logging.error(f"Aborting due to errors when running command {cmd}")
            sys.exit(1)

        cmd = ["cmake", "--build", "."] + build_args
        try:
            subprocess.check_call(cmd, cwd=self.build_temp)
        except subprocess.CalledProcessError:
            logging.error(f"Aborting due to errors when running command {cmd}")
            sys.exit(1)

        print()  # Add an empty line for cleaner output


def main():
    """Build the C++ code."""
    # with open("./requirements.txt", "r") as f:
    #     requires = f.read().splitlines()
    setup(
        ext_modules=[CMakeExtension("nocturne", "./nocturne")],
        cmdclass=dict(build_ext=CMakeBuild),
    )


if __name__ == "__main__":
    main()
