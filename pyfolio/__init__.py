import warnings

from . import utils
from . import timeseries
from . import pos
from . import txn

from .tears import *  # noqa
from .plotting import *  # noqa

try:
    from . import bayesian
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)

__version__ = '0.3'
__all__ = ['utils', 'timeseries', 'pos', 'txn', 'bayesian']
