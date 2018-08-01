# What's New

These are new features and improvements of note in each release.

## v0.9.0 (Aug 1st, 2018)

### New features

 - Previously, `pyfolio` has required a benchmark, usually the U.S. market
   returns `SPY`. In order to provide support for international equities and
   alternative data sets, `pyfolio` is now completely independent of benchmarks.
   If a benchmark is passed, all benchmark-related analyses will be performed;
   if not, they will simply be skipped. By [George Ho](https://github.com/eigenfoo)
  - Performance attribution tearsheet [PR441](https://github.com/quantopian/pyfolio/pull/441), [PR433](https://github.com/quantopian/pyfolio/pull/433), [PR442](https://github.com/quantopian/pyfolio/pull/442). By [Vikram Narayan](https://github.com/vikram-narayan).
  - Improved implementation of `get_turnover` [PR332](https://github.com/quantopian/pyfolio/pull/432). By [Gus Gordon](https://github.com/gusgordon).
  - Users can now pass in extra rows (as a dict or OrderedDict) to display in the perf_stats table [PR445](https://github.com/quantopian/pyfolio/pull/445). By [Gus Gordon](https://github.com/gusgordon).

### Maintenance

 - Many features have been more extensively troubleshooted, maintained and
   tested. By [Ana Ruelas](https://github.com/ahgnaw) and [Vikram
   Narayan](https://github.com/vikram-narayan).
 - Various fixes to support pandas versions >= 0.18.1 [PR443](https://github.com/quantopian/pyfolio/pull/443). By [Andrew Daniels](https://github.com/yankees714).

## v0.8.0 (Aug 23rd, 2017)

This is a major release from `0.7.0`, and all users are recommended to upgrade.

### New features

 - Risk tear sheet: added a new tear sheet to analyze risk exposures to common
   factors (e.g. mean reversion and momentum), sector (e.g. Morningstar
   sectors), market cap and illiquid stocks. By [George
   Ho](https://github.com/eigenfoo).
 - Simple tear sheet: added a new tear sheet that presents only the most
   important plots in the full tear sheet, for a quick general overview of a
   portfolio's performance. By [George Ho](https://github.com/eigenfoo).
 - Performance attribution: added new table to do performance attribution
   analysis, such as the amount of returns attributable to common factors, and
   summary statistics such as the multi-factor alpha and multi-factor Sharpe
   ratio. By [Vikram Narayan](https://github.com/vikram-narayan).
 - Volatility plot: added a rolling annual volatility plot to the returns tear
   sheet. By [hkopp](https://github.com/hkopp).

### Bugfixes

 - Yahoo and pandas data-reader: fixed bug regarding Yahoo backend for market
   data and pandas data-reader. By [Thomas Wiecki](https://github.com/twiecki)
   and [Gus Gordon](https://github.com/gusgordon).
 - `empyrical` compatibility: removed `information_ratio` to remain compatible
   with `empyrical`. By [Thomas Wiecki](https://github.com/twiecki).
 - Fama-French rolling multivariate regression: fixed bug where the rolling
   Fama-French plot performed separate linear regressions instead of a
   multivariate regression. By [George Ho](https://github.com/eigenfoo).
 - Other minor bugfixes. By [Scott Sanderson](https://github.com/ssanderson),
   [Jonathan Ng](https://github.com/jonathanng),
   [SylvainDe](https://github.com/SylvainDe) and
   [mckelvin](https://github.com/mckelvin).

### Maintenance

 - Documentation: updated and improved `pyfolio` documentation and example
   Jupyter notebooks. By [George Ho](https://github.com/eigenfoo).
 - Data loader migration: all data loaders have been migrated from `pyfolio` to
   `empyrical`. By [James Christopher](https://github.com/jameschristopher).
 - Improved plotting style: fixed issues with formatting and presentation of
   plots. By [George Ho](https://github.com/eigenfoo).

## v0.7.0 (Jan 28th, 2017)

This is a major release from `0.6.0`, and all users are recommended to upgrade.

### New features

 - Adds a transaction timing plot, which gives insight into the strategies'
   trade times.
 - Adds a plot showing the number of longs and shorts held over time.
 - New round trips plot selects a sample of held positions (16 by default) and
   shows their round trips. This replaces the old round trip plot, which became
   unreadable for strategies that traded many positions.
 - Adds basic capability for analyzing intraday strategies. If a strategy makes
   a large amount of transactions relative to its end-of-day positions, then
   pyfolio will attempt to reconstruct the intraday positions, take the point of
   peak exposure to the market during each day, and plot that data with the
   positions tear sheet. By default pyfolio will automatically detect this, but
   the behavior can be changed by passing either `estimate_intraday=True` or
   `estimate_intraday=False` to the tear sheet functions ([see
   here](https://github.com/quantopian/pyfolio/blob/master/pyfolio/tears.py#L131)).
 - Now formats [zipline](https://github.com/quantopian/zipline) assets,
   displaying their ticker symbol.
 - Gross leverage is no longer required to be passed, and will now be calculated
   from the passed positions DataFrame.

### Bugfixes

 - Cone plotting location is now correct.
 - Adjust scaling of beta and Fama-French plots.
 - Removed multiple dependencies, some of which were previously unused.
 - Various text fixes.

## v0.6.0 (Oct 17, 2016)

This is a major new release from `0.5.1`. All users are recommended to upgrade.

### New features

* Computation of performance and risk measures has been split off into
  [`empyrical`](https://github.com/quantopian/empyrical). This allows
  [`Zipline`](https://zipline.io) and `pyfolio` to use the same code to
  calculate its risk statistics. By [Ana Ruelas](https://github.com/ahgnaw) and
  [Abhi Kalyan](https://github.com/abhijeetkalyan).
* New multistrike cone which redraws the cone when it crossed its initial bounds
  [PR310](https://github.com/quantopian/pyfolio/pull/310). By [Ana
  Ruelas](https://github.com/ahgnaw) and [Abhi
  Kalyan](https://github.com/abhijeetkalyan).

### Bugfixes

* Can use most recent PyMC3 now.
* Depends on seaborn 0.7.0 or later now
  [PR331](https://github.com/quantopian/pyfolio/pull/331).
* Disable buggy computation of round trips per day and per month
  [PR339](https://github.com/quantopian/pyfolio/pull/339).

## v0.5.1 (June 10, 2016)

This is a bugfix release from `0.5.0` with limited new functionality. All users are recommended to upgrade.

### New features

* OOS data is now overlaid on top of box plot
  [PR306](https://github.com/quantopian/pyfolio/pull/306) by [Ana
  Ruelas](https://github.com/ahgnaw)
* New logo [PR298](https://github.com/quantopian/pyfolio/pull/298) by [Taso
  Petridis](https://github.com/tasopetridis) and [Richard
  Frank](https://github.com/richafrank)
* Raw returns plot and cumulative log returns plot
  [PR294](https://github.com/quantopian/pyfolio/pull/294) by [Thomas
  Wiecki](https://github.com/twiecki)
* Net exposure line to the long/short exposure plot
  [PR301](https://github.com/quantopian/pyfolio/pull/301) by [Ana
  Ruelas](https://github.com/ahgnaw)

### Bugfixes

* Fix drawdown behavior and pandas exception in tear-sheet creation
  [PR297](https://github.com/quantopian/pyfolio/pull/297) by [Flavio
  Duarte](https://github.com/flaviodrt)

## v0.5.0 (April 21, 2016) -- Olympia

This is a major release from `0.4.0` that includes many new analyses and
features. We recommend that all users upgrade to this new version. Also update
your dependencies, specifically, `pandas>=0.18.0`, `seaborn>=0.6.0` and
`zipline>=0.8.4`.

### New features

* New capacity tear-sheet to assess how much capital can be traded on a strategy
  [PR284](https://github.com/quantopian/pyfolio/pull/284). [Andrew
  Campbell](https://github.com/a-campbell).
* Bootstrap analysis to assess uncertainty in performance metrics
  [PR261](https://github.com/quantopian/pyfolio/pull/261). [Thomas
  Wiecki](https://github.com/twiecki)
* Refactored round-trip analysis to be more general and have better output. Now
  does full portfolio reconstruction to match trades
  [PR293](https://github.com/quantopian/pyfolio/pull/293). [Thomas
  Wiecki](https://github.com/twiecki), [Andrew
  Campbell](https://github.com/a-campbell). See the
  [tutorial](http://quantopian.github.io/pyfolio/round_trip_example/) for more
  information.
* Prettier printing of tables in notebooks
  [PR289](https://github.com/quantopian/pyfolio/pull/289). [Thomas
  Wiecki](https://github.com/twiecki)
* Faster max-drawdown calculation
  [PR281](https://github.com/quantopian/pyfolio/pull/281). [Devin
  Stevenson](https://github.com/devinstevenson)
* New metrics tail-ratio and common sense ratio
  [PR276](https://github.com/quantopian/pyfolio/pull/276). [Thomas
  Wiecki](https://github.com/twiecki)
* Log-scaled cumulative returns plot and raw returns plot
  [PR294](https://github.com/quantopian/pyfolio/pull/294). [Thomas
  Wiecki](https://github.com/twiecki)

### Bug fixes
* Many depracation fixes for Pandas 0.18.0, seaborn 0.6.0, and zipline 0.8.4


## v0.4.0 (Dec 10, 2015)

This is a major release from 0.3.1 that includes new features and quite a few bug fixes. We recommend that all users upgrade to this new version.

### New features

* Round-trip analysis [PR210](https://github.com/quantopian/pyfolio/pull/210)
  Andrew, Thomas
* Improved cone to forecast returns that uses a bootstrap instead of linear
  forecasting [PR233](https://github.com/quantopian/pyfolio/pull/233) Andrew,
  Thomas
* Plot max and median long/short exposures
  [PR237](https://github.com/quantopian/pyfolio/pull/237) Andrew

### Bug fixes

* Sharpe ratio was calculated incorrectly
  [PR219](https://github.com/quantopian/pyfolio/pull/219) Thomas, Justin
* annual_return() now only computes CAGR in the correct way
  [PR234](https://github.com/quantopian/pyfolio/pull/234) Justin
* Cache SPY and Fama-French returns in home-directory instead of
  install-directory [PR241](https://github.com/quantopian/pyfolio/pull/241) Joe
* Remove data files from package
  [PR241](https://github.com/quantopian/pyfolio/pull/241) Joe
* Cast factor.name to str
  [PR223](https://github.com/quantopian/pyfolio/pull/223) Scotty
* Test all `create_*_tear_sheet` functions in all configurations
  [PR247](https://github.com/quantopian/pyfolio/pull/247) Thomas


## v0.3.1 (Nov 12, 2015)

This is a minor release from 0.3 that includes mostly bugfixes but also some new features. We recommend that all users upgrade to this new version.

### New features

* Add Information Ratio [PR194](https://github.com/quantopian/pyfolio/pull/194)
  by @MridulS
* Bayesian tear-sheet now accepts 'Fama-French' option to do Bayesian
  multivariate regression against Fama-French risk factors
  [PR200](https://github.com/quantopian/pyfolio/pull/200) by Shane Bussman
* Plotting of monthly returns
  [PR195](https://github.com/quantopian/pyfolio/pull/195)

### Bug fixes

* `pos.get_percent_alloc` was not handling short allocations correctly
  [PR201](https://github.com/quantopian/pyfolio/pull/201)
* UTC bug with cached Fama-French factors
  [commit](https://github.com/quantopian/pyfolio/commit/709553a55b5df7c908d17f443cb17b51854a65be)
* Sector map was not being passed from `create_returns_tearsheet`
  [commit](https://github.com/quantopian/pyfolio/commit/894b753e365f9cb4861ffca2ef214c5a64b2bef4)
* New sector mapping feature was not Python 3 compatible
  [PR201](https://github.com/quantopian/pyfolio/pull/201)


### Maintenance

* We now depend on pandas-datareader as the yahoo finance loaders from pandas
  will be deprecated [PR181](https://github.com/quantopian/pyfolio/pull/181) by
  @tswrightsandpointe

### Contributors

Besiders the core developers, we have seen an increase in outside contributions
which we greatly appreciate. Specifically, these people contributed to this
release:

* Shane Bussman
* @MridulS
* @YihaoLu
* @jkrauss82
* @tswrightsandpointe
* @cgdeboer


## v0.3 (Oct 23, 2015)

This is a major release from 0.2 that includes many exciting new features. We
recommend that all users upgrade to this new version.

### New features

* Sector exposures: sum positions by sector given a dictionary or series of
  symbol to sector mappings
  [PR166](https://github.com/quantopian/pyfolio/pull/166)
* Ability to make cones with multiple shades stdev regions
  [PR168](https://github.com/quantopian/pyfolio/pull/168)
* Slippage sweep: See how an algorithm performs with various levels of slippage
  [PR170](https://github.com/quantopian/pyfolio/pull/170)
* Stochastic volatility model in Bayesian tear sheet
  [PR174](https://github.com/quantopian/pyfolio/pull/174)
* Ability to suppress display of position information
  [PR177](https://github.com/quantopian/pyfolio/pull/177)

### Bug fixes

* Various fixes to make pyfolio pandas 0.17 compatible

## v0.2 (Oct 16, 2015)

This is a major release from 0.1 that includes mainly bugfixes and refactorings
but also some new features. We recommend that all users upgrade to this new
version.

### New features

* Volatility matched cumulative returns plot
  [PR126](https://github.com/quantopian/pyfolio/pull/126).
* Allow for different periodicity (annualization factors) in the annual_()
  methods [PR164](https://github.com/quantopian/pyfolio/pull/164).
* Users can supply their own interesting periods
  [PR163](https://github.com/quantopian/pyfolio/pull/163).
* Ability to weight a portfolio of holdings by a metric valued
  [PR161](https://github.com/quantopian/pyfolio/pull/161).

### Bug fixes

* Fix drawdown overlaps [PR150](https://github.com/quantopian/pyfolio/pull/150).
* Monthly returns distribution should not stack by year
  [PR162](https://github.com/quantopian/pyfolio/pull/162).
* Fix gross leverage [PR147](https://github.com/quantopian/pyfolio/pull/147)
