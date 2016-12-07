![pyfolio](https://media.quantopian.com/logos/open_source/pyfolio-logo-03.png "pyfolio")

# pyfolio

[![Join the chat at https://gitter.im/quantopian/pyfolio](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/quantopian/pyfolio?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![build status](https://travis-ci.org/quantopian/pyfolio.png?branch=master)](https://travis-ci.org/quantopian/pyfolio)

pyfolio is a Python library for performance and risk analysis of
financial portfolios developed by
[Quantopian Inc](https://www.quantopian.com). It works well with the
[Zipline](http://zipline.io) open source backtesting library.

At the core of pyfolio is a so-called tear sheet that consists of
various individual plots that provide a comprehensive image of the
performance of a trading algorithm. Here's an example tear sheet analyzing returns, which comes from the Zipline algorithm sample notebook:

![example tear 0](https://github.com/quantopian/pyfolio/raw/master/docs/example_tear_0.png "Example tear sheet created from a Zipline algo")
![example tear 1](https://github.com/quantopian/pyfolio/raw/master/docs/example_tear_1.png "Example tear sheet created from a Zipline algo")

See also [slides of a talk about pyfolio.](http://nbviewer.ipython.org/format/slides/github/quantopian/pyfolio/blob/master/pyfolio/examples/overview_slides.ipynb#/)

## Installation

### (Optional) Virtual Environment

For development on pyfolio itself, you might want to use a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to avoid dependency conflicts between `pyfolio` and other python projects you have. To get set up with a virtual env, run:
```bash

mkvirtualenv pyfolio
```
before running the install commands below.

To install `pyfolio` via `pip` issue the following command:

```bash
pip install pyfolio
```

For development, clone the git repo and run `python setup.py develop`
and edit the library files directly. Make sure to reload or restart
the IPython kernel when you make changes.

`pyfolio` has the following dependencies:
 - numpy
 - scipy
 - pandas
 - matplotlib
 - [seaborn](https://github.com/mwaskom/seaborn)
 - [pymc3](https://github.com/pymc-devs/pymc3) (optional)
 - [zipline](https://github.com/quantopian/zipline) (optional; requires master, *not* 0.7.0)

Some of Pyfolio's functionality, such as the [Bayesian tearsheet](https://github.com/quantopian/pyfolio/blob/master/pyfolio/examples/bayesian.ipynb), requires PyMC3 and Theano. To get set up, you can run:

```bash
pip install theano
```

```bash
pip install git+https://github.com/pymc-devs/pymc3
```

If you are on OSX and using a non-framework build of python you may need to set your backend:
``` bash
echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc
```

## Usage

A good way to get started is to run the examples in a [Jupyter notebook](http://jupyter.org/).

To get set up with an example, you can:

Run a Jupyter notebook server via:

```bash
jupyter notebook
```

From the notebook list page(usually found at `http://localhost:8888/`), navigate over to the examples directory, and open any file with a .ipynb extension.

Execute the code in a notebook cell by clicking on it and hitting Shift+Enter.

## Questions?

If you find a bug, feel free to open an issue on our github tracker.

You can also join our [mailing list](https://groups.google.com/forum/#!forum/pyfolio).

## Contribute

If you want to contribute, a great place to start would be the [help-wanted issues](https://github.com/quantopian/pyfolio/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Credits

* Gus Gordon (gus@quantopian.com)
* Justin Lent (justin@quantopian.com)
* Sepideh Sadeghi (sp.sadeghi@gmail.com)
* Thomas Wiecki (thomas@quantopian.com)
* Jessica Stauth (jstauth@quantopian.com)
* Karen Rubin (karen@quantopian.com)
* David Edwards (dedwards@quantopian.com)
* Andrew Campbell (andrew@quantopian.com)

For a full list of contributors, see https://github.com/quantopian/pyfolio/graphs/contributors.
