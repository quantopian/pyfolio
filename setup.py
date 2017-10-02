#!/usr/bin/env python
import sys

from setuptools import setup

import versioneer

DISTNAME = 'pyfolio'
DESCRIPTION = "pyfolio is a Python library for performance and risk analysis of financial portfolios"
LONG_DESCRIPTION = """pyfolio is a Python library for performance and risk analysis of
financial portfolios developed by `Quantopian Inc`_. It works well with the
`Zipline`_ open source backtesting library.

At the core of pyfolio is a so-called tear sheet that consists of
various individual plots that provide a comprehensive performance
overview of a portfolio.

.. _Quantopian Inc: https://www.quantopian.com
.. _Zipline: http://zipline.io
"""
MAINTAINER = 'Quantopian Inc'
MAINTAINER_EMAIL = 'opensource@quantopian.com'
AUTHOR = 'Quantopian Inc'
AUTHOR_EMAIL = 'opensource@quantopian.com'
URL = "https://github.com/quantopian/pyfolio"
LICENSE = "Apache License, Version 2.0"
VERSION = "0.8.0"

classifiers = ['Development Status :: 4 - Beta',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

if (sys.version_info.major, sys.version_info.minor) >= (3, 3):
    support_ipython_6 = True
else:
    support_ipython_6 = False

install_reqs = [
    'ipython>=3.2.3' if support_ipython_6 else 'ipython>=3.2.3, <6',
    'matplotlib>=1.4.0',
    'numpy>=1.9.1',
    'pandas>=0.18.1',
    'pytz>=2014.10',
    'scipy>=0.14.0',
    'scikit-learn>=0.18.2',
    'seaborn>=0.7.1',
    'pandas-datareader>=0.2',
    'empyrical>=0.3.2'
]

test_reqs = ['nose>=1.3.7', 'nose-parameterized>=0.5.0', 'runipy>=0.1.3']
bayesian_reqs = ['pymc3 >= 3.1']

extras_reqs = {
    'bayesian': bayesian_reqs,
    'test': test_reqs,
    'all': test_reqs + bayesian_reqs,
}

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        cmdclass=versioneer.get_cmdclass(),
        version=versioneer.get_version(),
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=['pyfolio', 'pyfolio.tests'],
        package_data={'pyfolio': ['data/*.*']},
        classifiers=classifiers,
        install_requires=install_reqs,
        extras_require=extras_reqs,
        tests_require=test_reqs,
        test_suite='nose.collector',
    )
