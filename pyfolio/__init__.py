import warnings

from . import utils
from . import timeseries
from . import pos
from . import txn

from .tears import *  # noqa
from .plotting import *  # noqa
from ._version import get_versions

try:
    from . import bayesian
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)


__version__ = get_versions()['version']
del get_versions

__all__ = ['utils', 'timeseries', 'pos', 'txn', 'bayesian']
