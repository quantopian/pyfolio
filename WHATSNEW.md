# What's New

These are new features and improvements of note in each release.

## v0.3 (TBA)

### New features

* Sector exposures: sum positions by sector given a dictionary or series of symbol to sector mappings [PR166](https://github.com/quantopian/pyfolio/pull/166)

## v0.2 (Oct 16, 2015)

This is a major release from 0.1 that includes mainly bugfixes and refactorings but also some new features. We recommend that all users upgrade to this new version.

### New features

* Volatility matched cumulative returns plot [PR126](https://github.com/quantopian/pyfolio/pull/126).
* Allow for different periodicity (annualization factors) in the annual_() methods [PR164](https://github.com/quantopian/pyfolio/pull/164).
* Users can supply their own interesting periods [PR163](https://github.com/quantopian/pyfolio/pull/163).
* Ability to weight a portfolio of holdings by a metric valued [PR161](https://github.com/quantopian/pyfolio/pull/161).

### Bug fixes

* Fix drawdown overlaps [PR150](https://github.com/quantopian/pyfolio/pull/150).
* Monthly returns distribution should not stack by year [PR162](https://github.com/quantopian/pyfolio/pull/162).
* Fix gross leverage [PR147](https://github.com/quantopian/pyfolio/pull/147)
