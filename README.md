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

To install pyfolio, run:

```bash
pip install pyfolio
```

#### For development

For developing pyfolio, you may want to use a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to avoid dependency conflicts between pyfolio and other python projects you have. To get set up with a virtual env, run:
```bash
mkvirtualenv pyfolio
```

Next, clone this git repository and run `python setup.py develop`
and edit the library files directly.


#### Bayesian tear sheet

Generating a [Bayesian tearsheet](https://github.com/quantopian/pyfolio/blob/master/pyfolio/examples/bayesian.ipynb) requires PyMC3 and Theano. You can install these packages with the following commands:

```bash
pip install theano
```

```bash
pip install pymc3
```

#### Matplotlib on OSX

If you are on OSX and using a non-framework build of python you may need to set your backend:
``` bash
echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc
```

## Usage

A good way to get started is to run the examples in a [Jupyter notebook](http://jupyter.org/).

To start with an example, you can run a Jupyter notebook server:

```bash
jupyter notebook
```

From the notebook list page, navigate to the pyfolio examples directory,
and open a notebook. Execute the code in a notebook cell byclicking on it
and hitting Shift+Enter.


## Questions?

If you find a bug, feel free to open an issue in this repository.

You can also join our [mailing list](https://groups.google.com/forum/#!forum/pyfolio) or
our [Gitter channel](https://gitter.im/quantopian/pyfolio).


## Contributing

If you'd like to contribute, a great place to look is the [issues marked with help-wanted](https://github.com/quantopian/pyfolio/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For a list of contributors, see https://github.com/quantopian/pyfolio/graphs/contributors.
