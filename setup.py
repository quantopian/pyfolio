#!/usr/bin/env python
from setuptools import setup
import sys


DISTNAME = 'quantrisk'
DESCRIPTION = "QuantRisk"
LONG_DESCRIPTION    = """"""
MAINTAINER = 'Quantopian Inc'
MAINTAINER_EMAIL = 'opensource@quantopian.com'
AUTHOR = 'Quantopian Inc'
AUTHOR_EMAIL = 'opensource@quantopian.com'
URL = "https://github.com/quantopian/quantrisk"
LICENSE = "Apache License, Version 2.0"
VERSION = "0.1"

classifiers = ['Development Status :: 3 - Alpha',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

install_reqs = ['numpy>=1.9.2', 'scipy>=0.13.0', 'matplotlib>=1.2.1',
                'seaborn>=0.5.1', 'pandas>=0.16.1', 'statsmodels>=0.6.1', 'zipline>=0.7.0']

test_reqs = ['nose']

dep_links = []

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          packages=['quantrisk', 'quantrisk.tests'],
          package_data = {'quantrisk.examples': ['data/*.*']},
          classifiers=classifiers,
          install_requires=install_reqs,
          dependency_links=dep_links,
          tests_require=test_reqs,
          test_suite='nose.collector')
