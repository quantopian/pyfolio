#!/usr/bin/env python
"""
simple example script for running notebooks and reporting exceptions.
Usage: `checkipnb.py foo.ipynb [bar.ipynb [...]]`
Each cell is submitted to the kernel, and checked for errors.
"""

from pathlib import Path
from runipy.notebook_runner import NotebookRunner
from pyfolio.ipycompat import read as read_notebook

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "examples"


def test_nbs():
    for ipynb in EXAMPLES_PATH.glob("*.ipynb"):
        print(ipynb)
        with open(ipynb) as f:
            nb = read_notebook(f, "json")
            nb_runner = NotebookRunner(nb)
            nb_runner.run_notebook(skip_exceptions=False)
