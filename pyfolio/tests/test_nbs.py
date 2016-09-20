import os
import glob
from warnings import warn

from runipy.notebook_runner import NotebookRunner
from IPython.nbformat import read

import pyfolio


pyfolio_root = os.path.dirname(pyfolio.__file__)


def test_nbs():
    path = os.path.join(pyfolio_root, 'examples', '*.ipynb')
    for ipynb in glob.glob(path):

        # See if bayesian is useable before we run a test
        if ipynb.endswith('bayesian.ipynb'):
            try:
                import pymc3  # NOQA
            except:
                yield warn, 'skipping test nb %s because missing pymc3' % ipynb
                continue

        with open(ipynb) as f:
            nb = read(f, 'json')
            nb_runner = NotebookRunner(nb)
            yield nb_runner.run_notebook
