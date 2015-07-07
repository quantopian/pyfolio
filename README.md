# QuantRisk

QuantRisk is a Python library for performance and risk analysis of
financial portfolios developed by
[Quantopian Inc](https://www.quantopian.com). It works well with the
[Zipline](http://zipline.io) open source backtesting library.

At the core of QuantRisk is a so-called tear sheet that consists of
various individual plots that provide a comprehensive image of the
performance of a trading algorithm. Here is an example of a tear sheet of a returns-based analysis of the `$FB` stock:

![example returns](https://github.com/quantopian/quantrisk/raw/master/docs/example_returns.png "Example tear sheet about $FB stock")

## Installation

To install `quantrisk` via `pip` issue the following command:

```bash
pip install --pre quantrisk
```

For development, clone the git repo and run `python setup.py develop`
and edit the library files directly. Make sure to reload or restart
the IPython kernel when you make changes.

`quantrisk` has the following dependencies:
* numpy
* scipy
* pandas
* matplotlib
* [seaborn](https://github.com/mwaskom/seaborn)
* [pymc3](https://github.com/pymc-devs/pymc3) (optional)

## Questions?

If you find a bug, feel free to open an issue on our github tracker.

You can also join our [mailing list](https://groups.google.com/forum/#!forum/quantrisk).

## Credits

* Gus Gordon
* Justin Lent
* Sepideh Saghedi
* Thomas Wiecki
