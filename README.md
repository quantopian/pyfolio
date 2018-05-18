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
performance of a trading algorithm. Here's an example of a simple tear
sheet analyzing a strategy:

![simple tear 0](https://github.com/quantopian/pyfolio/raw/master/docs/simple_tear_0.png "Example tear sheet created from a Zipline algo")
![simple tear 1](https://github.com/quantopian/pyfolio/raw/master/docs/simple_tear_1.png "Example tear sheet created from a Zipline algo")

Also see [slides of a talk about
pyfolio](http://nbviewer.jupyter.org/format/slides/github/quantopian/pyfolio/blob/master/pyfolio/examples/pyfolio_talk_slides.ipynb#/).

## Installation

To install pyfolio, run:

```bash
pip install pyfolio
```

#### Development

For development, you may want to use a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to avoid dependency conflicts between pyfolio and other Python projects you have. To get set up with a virtual env, run:
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

If you are on OSX and using a non-framework build of Python, you may need to set your backend:
``` bash
echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc
```

## Usage

A good way to get started is to run the pyfolio examples in
a [Jupyter notebook](http://jupyter.org/). To do this, you first want to
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

### Deprecated: Data Reading via `pandas-datareader`

As of early 2018, Yahoo Finance has suffered major API breaks with no stable
replacement, and the Google Finance API has not been stable since late 2017
[(source)](https://github.com/pydata/pandas-datareader/blob/da18fbd7621d473828d7fa81dfa5e0f9516b6793/README.rst).
In recent months it has become a greater and greater strain on the `empyrical`
and `pyfolio` development teams to maintain support for fetching data through
`pandas-datareader` and other third-party libraries, as these APIs are known to
be unstable.

As a result, all `empyrical` (and therefore `pyfolio`, which is a downstream
dependency) support for data reading functionality has been deprecated and will
be removed in a future version.

Users should beware that the following functions are now deprecated:

- `pyfolio.utils.default_returns_func`
- `pyfolio.utils.get_fama_french`
- `pyfolio.utils.get_returns_cached`
- `pyfolio.utils.get_symbol_returns_from_yahoo`
- `pyfolio.utils.get_treasury_yield`
- `pyfolio.utils.get_utc_timestamp`
- `pyfolio.utils.cache_dir`
- `pyfolio.utils.ensure_directory`
- `pyfolio.utils.data_path`
- `pyfolio.utils._1_bday_ago`
- `pyfolio.utils.load_portfolio_risk_factors`

Users should expect regular failures from the following functions, pending
patches to the Yahoo or Google Finance API:

- `pyfolio.utils.default_returns_func`
- `pyfolio.utils.get_symbol_returns_from_yahoo`

For alternative data sources, we suggest the following:

1. Migrate your research workflow to the Quantopian Research environment,
   where there is [free and flexible data access to over 57
   datasets](https://www.quantopian.com/data)
2. Make use of any remaining functional APIs supported by
   `pandas-datareader`. These include:

   - [Morningstar](https://pydata.github.io/pandas-datareader/stable/remote_data.html#remote-data-morningstar)
   - [Quandl](https://pydata.github.io/pandas-datareader/stable/remote_data.html#remote-data-quandl)

   Please note that you may need to create free accounts with these data
   providers and receive an API key in order to access data. These API keys
   should be set as environment variables, or passed as an argument to
   `pandas-datareader`.


## Contributing

If you'd like to contribute, a great place to look is the [issues marked with help-wanted](https://github.com/quantopian/pyfolio/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For a list of core developers and outside collaborators, see [the GitHub contributors list](https://github.com/quantopian/pyfolio/graphs/contributors).
