#
# Copyright 2017 Quantopian, Inc.
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
from __future__ import division

import engarde.checks as ed


class TearSheetInputError(Exception):
    """Custom Exception for pyfolio input errors"""
    pass


def check_inputs(returns=None,
                 positions=None,
                 txns=None):
    """
    Check inputs to create_foo_tear_sheet functions

    - Raises a TearSheetInputError if inputs violate specification
    """
    if returns is not None:
        check_returns(returns)

    if positions is not None:
        check_positions(positions)

    if txns is not None:
        check_txns(txns)


def check_returns(returns):
    """
    Check returns for specification

    - There must not be NaNs
    - Index must be timezone-localized datetimes
    - Returns must be floats
    """
    try:
        ed.none_missing(returns)
    except AssertionError as e:
        raise TearSheetInputError('''returns has NaNs. engarde raises the
                                  following: ''' + e)

    if returns.index.tz is None:
        raise TearSheetInputError('returns index is not timezone-localized')

    if returns.dtype != float:
        raise TearSheetInputError('returns data types are not all float')


def check_positions(positions):
    """
    Check positions for specification

    - There must not be any NaNs
    - Index must be timezone-localized datetimes
    - There must a column called 'cash'
    - Positions must be floats
    """
    try:
        ed.none_missing(positions)
    except AssertionError as e:
        raise TearSheetInputError('''positions has NaNs. engarde raises the
                                  following: ''' + e)

    if positions.index.tz is None:
        raise TearSheetInputError('positions index is not timezone-localized')

    if 'cash' not in positions.columns:
        raise TearSheetInputError('positions does not have a cash column')

    if not all(
        positions.apply(
            lambda col: col.dtype,
            axis='columns') == float):
        raise TearSheetInputError('''positions data types are not all float''')


def check_txns(txns):
    """
    Check transactions for specification

    - There must not be any NaNs
    - Index must be timezone-localized datetimes
    - There must be columns called 'amount', 'price', and 'symbol'
    - Amounts must be ints
    - Prices must be floats
    - Symbols must be strings
    """
    try:
        ed.none_missing(txns)
    except AssertionError as e:
        raise TearSheetInputError('''transactions has NaNs. engarde raises the
                                  following: ''' + e)

    if txns.index.tz is None:
        raise TearSheetInputError('''transactions index is not
                                  timezone-localized''')

    if not all([col in txns.columns for col in ['amount',
                                                'price',
                                                'symbol']]):
        raise TearSheetInputError('''transactions does not have correct columns.
                                  Columns must be "amount", "price" and
                                  "symbol"''')

    try:
        ed.has_dtypes(txns, {'amount': int,
                             'price': float,
                             'symbol': str})
    except AssertionError as e:
        raise TearSheetInputError('''transactions columns do not have correct
                                  data types. engarde raises the following: '''
                                  + e)
