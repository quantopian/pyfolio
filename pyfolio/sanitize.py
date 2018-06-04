#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from warnings import warn
import pandas as pd


def sanitize(returns=None,
             positions=None,
             txns=None):
    """
    Sanitize inputs to pyfolio.

    Returns
    -------
    returns, positions, txns : pd.Series or pd.DataFrame
        Sanitized inputs to pyfolio create_x_tear_sheet functions.
        See pyfolio.create_full_tear_sheet for more details.

    Usage
    -----
    To sanitize all inputs:
        rets, pos, txns = sanitize_inputs(rets, pos, txns)
    To sanitize only e.g. positions:
        _, pos, _ = sanitize_inputs(positions=pos)
    """
    if returns is not None:
        returns = sanitize_returns(returns)

    if positions is not None:
        positions = sanitize_positions(positions)

    if txns is not None:
        txns = sanitize_txns(txns)

    return returns, positions, txns


def sanitize_returns(returns):
    """
    Sanitize returns.

    Returns
    -------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

    Notes
    -----
    - Must be a pd.Series
        - If it is a 1D dataframe, squeeze and warn.
        - Otherwise, raise a ValueError
    - There must not be any NaNs
        - If there are, raise a ValueError
    - Index must be timezone-localized datetimes
        - If not, localize to UTC and warn.
    - Returns must be floats
        - If not, coerce values to float and warn
        - Failing that, raise a ValueError
    """
    try:
        assert isinstance(returns, pd.Series)
    except AssertionError:
        if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
            returns = returns.squeeze()
            msg = ('`returns` is a 1-dimensional pd.DataFrame. '
                   'Squeezing into a pd.Series...')
            warn(msg)
        else:
            msg = ('`returns` is not a pd.Series, and could not be coerced '
                   'into one.')
            raise ValueError(msg)

    try:
        assert not returns.isnull().any()
    except AssertionError:
        msg = ('`returns` has NaN values. Perhaps those are days with zero '
               'returns, or are not trading days?')
        raise ValueError(msg)

    try:
        assert returns.index.tz is not None
    except AssertionError:
        returns.index = returns.index.tz_localize('UTC')
        msg = ('`returns` index is not timezone-localized. '
               'Localizing to UTC...')
        warn(msg)

    try:
        assert returns.dtype == float
    except AssertionError:
        try:
            returns = returns.astype(float)
            msg = '`returns` does not have float dtype. Coercing to float...'
            warn(msg)
        except ValueError:
            msg = ('`returns` does not have float dtype, and could not be '
                   'coerced into floats.')
            raise ValueError(msg)

    return returns


def sanitize_positions(positions):
    """
    Sanitize positions.

    Returns
    -------
    sanitized_positions : pd.DataFrame
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375

    Notes
    -----
    - Must be a pd.DataFrame
        - If not, raise a ValueError
    - There must not be any NaNs
        - If there are, fill NaNs with zeros and warn
    - Index must be timezone-localized datetimes
        - If not, localize to UTC and warn
    - There must a column called 'cash'
        - If not, raise ValueError
    - Positions must be floats
        - if not, coerce values to float and warn
        - Failing that, raise a ValueError
    """
    try:
        assert isinstance(positions, pd.DataFrame)
    except AssertionError:
        msg = '`positions` is not a pd.DataFrame.'
        raise ValueError(msg)

    try:
        assert not positions.isnull().any().any()
    except AssertionError:
        positions = positions.fillna(0)
        msg = '`positions` contains NaNs. Filling with 0...'
        warn(msg)

    try:
        assert positions.index.tz is not None
    except AssertionError:
        positions.index = positions.index.tz_localize('UTC')
        msg = ('`positions` index is not timezone-localized. '
               'Localizing to UTC...')
        warn(msg)

    try:
        assert 'cash' in positions.columns
    except AssertionError:
        msg = '`positions does not contain a `cash` column.'
        raise ValueError(msg)

    try:
        # FIXME more idiomatic way to check dtype of all columns?
        assert all(positions.apply(lambda col: col.dtype,
                                   axis='columns') == float)
    except AssertionError:
        try:
            positions = positions.astype(float)
            msg = '`positions` does not have float dtype. Coercing to float...'
            warn(msg)
        except ValueError:
            msg = ('`positions` does not have float dtype, and could not be '
                   'coerced into floats.')
            raise ValueError(msg)

    return positions


def sanitize_txns(txns):
    """
    Sanitize transactions.

    Returns
    -------
    txns : pd.DataFrame
        Executed trade volumes and fill prices.
        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'

    Notes
    -----
    - Must be a DataFrame
        - If not, raise ValueError
    - There must not be any NaNs
        - If there are, raise a ValueError
    - Index must be timezone-localized datetimes
        - If not, localize to UTC and warn
    - There must be columns called 'amount', 'price', and 'symbol'
        - If not, raise a ValueError
    - Amounts must be ints
    - Prices must be floats
    - Symbols must be strings or ints
        - If not, warn
    """
    try:
        assert isinstance(txns, pd.DataFrame)
    except AssertionError:
        msg = '`txns` is not a pd.DataFrame.'
        raise ValueError(msg)

    try:
        assert not txns.isnull().any().any()
    except AssertionError:
        msg = '`txns` contains NaNs.'
        raise ValueError(msg)

    try:
        assert txns.index.tz is not None
    except AssertionError:
        txns.index = txns.index.tz_localize('UTC')
        msg = ('`txns` index is not timezone-localized. '
               'Localizing to UTC...')
        warn(msg)

    try:
        assert set(['amount', 'price', 'symbol']) <= set(txns.columns)
    except AssertionError:
        msg = '`txns` does not have an `amount`, `price`, or `symbol` column.'
        raise ValueError(msg)

    try:
        assert txns.loc[:, 'amount'].dtype == int
        assert txns.loc[:, 'price'].dtype == float
        assert txns.loc[:, 'symbol'].dtype in [str, int]
    except AssertionError:
        msg = ('`txns` columns do not have the correct dtypes. '
               '`amount` column should have int dtype, '
               '`price` column should have float dtype, and '
               '`symbol` column should have str or int dtype. '
               'Attempting to create tear sheets...')
        warn(msg)

    return txns
