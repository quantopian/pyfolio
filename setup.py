#!/usr/bin/env python
from setuptools import setup
import sys


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
VERSION = "0.3"

classifiers = ['Development Status :: 4 - Beta',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.4',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

install_reqs = [
    'funcsigs>=0.4',
    'matplotlib>=1.4.0',
    'mock>=1.1.2',
    'numpy>=1.9.1',
    'pandas>=0.15.0',
    'pyparsing>=2.0.3',
    'python-dateutil>=2.4.2',
    'pytz>=2014.10',
    'scikit-learn>=0.15.0',
    'scipy>=0.14.0',
    'seaborn>=0.6.0',
    'statsmodels>=0.5.0']

extras_reqs = {
    'bayesian': ['pymc3']
}

test_reqs = ['nose>=1.3.7', 'nose-parameterized>=0.5.0', 'runipy>=0.1.3']

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
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
          extras_requires=extras_reqs,
          tests_require=test_reqs,
          test_suite='nose.collector')
