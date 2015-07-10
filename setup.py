#!/usr/bin/env python
from setuptools import setup
import sys


DISTNAME = 'pyfolio'
DESCRIPTION = "pyfolio is a Python library for performance and risk analysis of
financial portfolios"
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

install_reqs = ['numpy>=1.9.2', 'scipy>=0.13.0', 'matplotlib>=1.2.1',
                'seaborn>=0.5.1', 'pandas>=0.16.1', 'statsmodels>=0.6.1']

test_reqs = ['nose']

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
          package_data = {'pyfolio': ['data/*.*']},
          classifiers=classifiers,
          #install_requires=install_reqs,
          tests_require=test_reqs,
          test_suite='nose.collector')
