#!/usr/bin/env python
from setuptools import setup
import sys


DISTNAME = 'pyfolio'
DESCRIPTION = "pyfolio is a Python library for performance and risk analysis of financial portfolios"
LONG_DESCRIPTION = """pyfolio is a Python library for performance and risk analysis of
financial portfolios."""
MAINTAINER = 'Quantopian Inc'
MAINTAINER_EMAIL = 'opensource@quantopian.com'
AUTHOR = 'Quantopian Inc'
AUTHOR_EMAIL = 'opensource@quantopian.com'
URL = "https://github.com/quantopian/pyfolio"
LICENSE = "Apache License, Version 2.0"
VERSION = "0.1.beta"

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
    'Theano>=0.7.0',
    'funcsigs>=0.4',
    'matplotlib>=1.4.3',
    'mock>=1.1.2',
    'numpy>=1.9.2',
    'pandas>=0.16.2',
    'patsy>=0.3.0',
    'pbr>=1.3.0',
    'pyfolio>=0.1.beta',
    'pymc3>=3.0',
    'pyparsing>=2.0.3',
    'python-dateutil>=2.4.2',
    'pytz>=2015.4',
    'scikit-learn>=0.16.1',
    'scipy>=0.15.1',
    'seaborn>=0.6.0',
    'six>=1.9.0',
    'statsmodels>=0.6.1',
    'wsgiref>=0.1.2']

test_reqs = ['nose>=1.3.7', 'nose-parameterized>=0.5.0']

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
          tests_require=test_reqs,
          test_suite='nose.collector')
