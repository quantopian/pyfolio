#!/usr/bin/env python
import sys
from pathlib import Path
from setuptools import setup

# ensure the current directory is on sys.path
# so versioneer can be imported when pip uses
# PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(Path(__file__).resolve(strict=True).parent.as_posix())
import versioneer  # noqa: E402


if __name__ == "__main__":
    setup(
        cmdclass=versioneer.get_cmdclass(),
        version=versioneer.get_version(),
        packages=["pyfolio", "pyfolio.tests"],
        package_data={"pyfolio": ["data/*.*"]},
        test_suite="nose.collector",
    )
