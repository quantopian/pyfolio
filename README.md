![pyfolio](https://media.quantopian.com/logos/open_source/pyfolio-logo-03.png "pyfolio")

# pyfolio

[![Join the chat at https://gitter.im/quantopian/pyfolio](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/quantopian/pyfolio?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![build status](https://travis-ci.org/quantopian/pyfolio.png?branch=master)](https://travis-ci.org/quantopian/pyfolio)

pyfolio is a Python library for performance and risk analysis of
financial portfolios developed by
[Quantopian Inc](https://www.quantopian.com). It works well with the
[Zipline](https://www.zipline.io/) open source backtesting library.
Quantopian also offers a [fully managed service for professionals](https://factset.quantopian.com) 
that includes Zipline, Alphalens, Pyfolio, FactSet data, and more.

At the core of pyfolio is a so-called tear sheet that consists of
various individual plots that provide a comprehensive image of the
performance of a trading algorithm. Here's an example of a simple tear
sheet analyzing a strategy:

![simple tear 0](https://github.com/quantopian/pyfolio/raw/master/docs/simple_tear_0.png "Example tear sheet created from a Zipline algo")
![simple tear 1](https://github.com/quantopian/pyfolio/raw/master/docs/simple_tear_1.png "Example tear sheet created from a Zipline algo")

Also see [slides of a talk about
pyfolio](https://nbviewer.jupyter.org/format/slides/github/quantopian/pyfolio/blob/master/pyfolio/examples/pyfolio_talk_slides.ipynb#/).

## Installation

To install pyfolio, run:

```bash
pip install pyfolio
```

#### Development

For development, you may want to use a [virtual environment](https://docs.python-guide.org/en/latest/dev/virtualenvs/) to avoid dependency conflicts between pyfolio and other Python projects you have. To get set up with a virtual env, run:
```bash
mkvirtualenv pyfolio
```

Next, clone this git repository and run `python setup.py develop`
and edit the library files directly.

#### Matplotlib on OSX

If you are on OSX and using a non-framework build of Python, you may need to set your backend:
``` bash
echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc
```

## Usage

A good way to get started is to run the pyfolio examples in
a [Jupyter notebook](https://jupyter.org/). To do this, you first want to
start a Jupyter notebook server:

```bash
jupyter notebook
```

From the notebook list page, navigate to the pyfolio examples directory
and open a notebook. Execute the code in a notebook cell by clicking on it
and hitting Shift+Enter.


## Questions?

If you find a bug, feel free to [open an issue](https://github.com/quantopian/pyfolio/issues) in this repository.

You can also join our [mailing list](https://groups.google.com/forum/#!forum/pyfolio) or
our [Gitter channel](https://gitter.im/quantopian/pyfolio).

## Support

Please [open an issue](https://github.com/quantopian/pyfolio/issues/new) for support.

## Contributing

If you'd like to contribute, a great place to look is the [issues marked with help-wanted](https://github.com/quantopian/pyfolio/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For a list of core developers and outside collaborators, see [the GitHub contributors list](https://github.com/quantopian/pyfolio/graphs/contributors).
