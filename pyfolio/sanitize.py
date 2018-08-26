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
    if not isinstance(returns, pd.Series):
        if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
            sanitized_returns = sanitized_returns.squeeze()
            msg = ('`returns` is a 1-dimensional pd.DataFrame. '
                   'Squeezing into a pd.Series...')
            warn(msg)
        else:
            msg = ('`returns` is not a pd.Series, and could not be coerced '
                   'into one.')
            raise ValueError(msg)

    sanitized_returns = returns.copy()

    if sanitized_returns.isnull().any():
        msg = ('`returns` has NaN values. Perhaps those are days with zero '
               'returns, or are not trading days?')
        raise ValueError(msg)

    if sanitized_returns.index.tz is None:
        sanitized_returns.index = sanitized_returns.index.tz_localize('UTC')
        msg = ('`returns` index is not timezone-localized. '
               'Localizing to UTC...')
        warn(msg)

    if sanitized_returns.dtype != float:
        try:
            sanitized_returns = sanitized_returns.astype(float)
            msg = '`returns` does not have float dtype. Coercing to float...'
            warn(msg)
        except ValueError:
            msg = ('`returns` does not have float dtype, and could not be '
                   'coerced into floats.')
            raise ValueError(msg)

    return sanitized_returns


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
    if not isinstance(positions, pd.DataFrame):
        msg = '`positions` is not a pd.DataFrame.'
        raise ValueError(msg)

    sanitized_positions = positions.copy()

    if sanitized_positions.isnull().any().any():
        sanitized_positions = sanitized_positions.fillna(0)
        msg = '`positions` contains NaNs. Filling with 0...'
        warn(msg)

    if sanitized_positions.index.tz is None:
        sanitized_positions.index = sanitized_positions.index.tz_localize('UTC')
        msg = ('`positions` index is not timezone-localized. '
               'Localizing to UTC...')
        warn(msg)

    if 'cash' not in sanitized_positions.columns:
        msg = '`positions does not contain a `cash` column.'
        raise ValueError(msg)

    if not (positions.dtypes == np.dtype('float64')).all():
        try:
            sanitized_positions = sanitized_positions.astype(float)
            msg = '`positions` does not have float dtype. Coercing to float...'
            warn(msg)
        except ValueError:
            msg = ('`positions` does not have float dtype, and could not be '
                   'coerced into floats.')
            raise ValueError(msg)

    return sanitized_positions


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
    if not isinstance(txns, pd.DataFrame):
        msg = '`txns` is not a pd.DataFrame.'
        raise ValueError(msg)

    sanitized_txns = txns.copy()

    if sanitized_txns.isnull().any().any():
        msg = '`txns` contains NaNs.'
        raise ValueError(msg)

    if sanitized_txns.index.tz is None:
        sanitized_txns.index = sanitized_txns.index.tz_localize('UTC')
        msg = ('`txns` index is not timezone-localized. '
               'Localizing to UTC...')
        warn(msg)

    if not set(['amount', 'price', 'symbol']) <= set(sanitized_txns.columns):
        msg = '`txns` does not have an `amount`, `price`, or `symbol` column.'
        raise ValueError(msg)

    if not (sanitized_txns.loc[:, 'amount'].dtype == int
            and sanitized_txns.loc[:, 'price'].dtype == float
            and sanitized_txns.loc[:, 'symbol'].dtype in [str, int]):
        msg = ('`txns` columns do not have the correct dtypes. '
               '`amount` column should have int dtype, '
               '`price` column should have float dtype, and '
               '`symbol` column should have str or int dtype. '
               'Attempting to create tear sheets...')
        warn(msg)

    return sanitized_txns
