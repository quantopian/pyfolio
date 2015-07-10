import warnings

import utils
import timeseries
import positions
import txn
import bayesian

from .tears import *
from .plotting import *

__version__ = '0.1.beta'

try:
    import bayesian
except ImportError:
    warnings.warn("Could not import bayesian submodule due to missing pymc3 dependency.", ImportWarning)
