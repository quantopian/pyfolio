import warnings

from . import utils
from . import timeseries
from . import pos
from . import txn
from . import bayesian

from .tears import *
from .plotting import *

__version__ = '0.1.beta'

try:
    from . import bayesian
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)
