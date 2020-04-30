"""
TechMiner.datasets
===============================================================================

Overview
-------------------------------------------------------------------------------

The functions in this module allows the user to load bibliographical datasets.


Functions in this module
-------------------------------------------------------------------------------


"""

import pandas as pd
from os.path import join, dirname


class _Bunch(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def load_dynacol():
    """Load and return the dynacol dataset.

    This dataset contains the bibliographical information about publications
    in Scopus of the Dyna-Colombia journal, edited by Facultad de Minas,
    Universidad Nacional de Colombia, Sede Medellin, between January, 2010
    and September, 2019.

    Args:
        None.

    Returns:
        A dictionary.

    **Examples**

    >>> from techminer.datasets import load_dynacol
    >>> data = load_dynacol()
    >>> data.data.info()  # doctest: +NORMALIZE_WHITESPACE
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1714 entries, 0 to 999
    Data columns (total 14 columns):
     #   Column                         Non-Null Count  Dtype
    ---  ------                         --------------  -----
     0   Abstract                       1714 non-null   object
     1   Affiliations                   1706 non-null   object
     2   Author Keywords                1681 non-null   object
     3   Author(s) ID                   1714 non-null   object
     4   Authors                        1714 non-null   object
     5   Cited by                       1714 non-null   int64
     6   Issue                          1714 non-null   object
     7   Language of Original Document  1714 non-null   object
     8   Page end                       1669 non-null   float64
     9   Page start                     1677 non-null   float64
     10  Title                          1714 non-null   object
     11  Volume                         1714 non-null   int64
     12  Year                           1714 non-null   int64
     13  References                     1694 non-null   object
    dtypes: float64(2), int64(3), object(9)
    memory usage: 200.9+ KB


    """

    module_path = dirname(__file__)

    with open(join(module_path, "datasets/dyna-col.rst")) as rst_file:
        fdescr = rst_file.read()

    fdata = pd.read_json(join(module_path, "datasets/dyna-col.json"), orient="index")

    return _Bunch(data=fdata, DESCR=fdescr)


def load_dynacol_citedby():
    """Load and return the dynacol-citedby dataset.

    This dataset contains the bibliographical information about publications
    citing in Scopus the Dyna-Colombia journal, edited by Facultad de Minas,
    Universidad Nacional de Colombia, Sede Medellin, between January, 2010
    and September, 2019.

    Args:
        None.

    Returns:
        A dictionary.

    **Examples**

    >>> from techminer.datasets import load_dynacol_citedby
    >>> data = load_dynacol_citedby().data
    >>> len(data)  # doctest: +NORMALIZE_WHITESPACE
    3411


    """

    module_path = dirname(__file__)

    with open(join(module_path, "datasets/dyna-col-citedby.rst")) as rst_file:
        fdescr = rst_file.read()

    fdata = pd.read_json(
        join(module_path, "datasets/dyna-col-citedby.json"), orient="index"
    )

    return _Bunch(data=fdata, DESCR=fdescr)


def load_autotrading_raw():
    """Load and return the autotrading dataset with raw info.

    This dataset contains the raw bibliographical information for publications
    in Scopus automatic trading.

    Args:
        None.

    Returns:
        A dictionary.

    **Examples**

    >>> from techminer.datasets import load_autotrading_raw
    >>> data = load_autotrading_raw()
    >>> data.data.info()  # doctest: +NORMALIZE_WHITESPACE
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 212 entries, 0 to 99
    Data columns (total 18 columns):
     #   Column           Non-Null Count  Dtype
    ---  ------           --------------  -----
     0   Abstract         212 non-null    object
     1   Affiliations     194 non-null    object
     2   Author Keywords  155 non-null    object
     3   Author(s) ID     212 non-null    object
     4   Authors          212 non-null    object
     5   Cited by         212 non-null    int64
     6   DOI              152 non-null    object
     7   Document Type    212 non-null    object
     8   EID              212 non-null    object
     9   Index Keywords   135 non-null    object
     10  Issue            83 non-null     object
     11  Page end         185 non-null    float64
     12  Page start       185 non-null    float64
     13  Selected         212 non-null    bool
     14  Source title     212 non-null    object
     15  Title            212 non-null    object
     16  Volume           141 non-null    object
     17  Year             212 non-null    int64
    dtypes: bool(1), float64(2), int64(2), object(13)
    memory usage: 30.0+ KB

    """

    module_path = dirname(__file__)

    with open(join(module_path, "datasets/auto-trading-raw.rst")) as rst_file:
        fdescr = rst_file.read()

    fdata = pd.read_json(
        join(module_path, "datasets/auto-trading-raw.json"), orient="index"
    )

    return _Bunch(data=fdata, DESCR=fdescr)


def load_autotrading_selected():
    """Load and return the autotrading dataset with selected info.

    This dataset contains the raw bibliographical information for publications
    in Scopus automatic trading.

    Args:
        None.

    Returns:
        A dictionary.

    **Examples**

    >>> from techminer.datasets import load_autotrading_selected
    >>> data = load_autotrading_selected()
    >>> data.data.info()  # doctest: +NORMALIZE_WHITESPACE
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 95 entries, 0 to 99
    Data columns (total 17 columns):
     #   Column           Non-Null Count  Dtype
    ---  ------           --------------  -----
     0   Abstract         95 non-null     object
     1   Affiliations     95 non-null     object
     2   Author Keywords  76 non-null     object
     3   Author(s) ID     95 non-null     object
     4   Authors          95 non-null     object
     5   Cited by         95 non-null     int64
     6   DOI              82 non-null     object
     7   Document Type    95 non-null     object
     8   EID              95 non-null     object
     9   Index Keywords   71 non-null     object
     10  Issue            33 non-null     object
     11  Page end         88 non-null     float64
     12  Page start       88 non-null     float64
     13  Source title     95 non-null     object
     14  Title            95 non-null     object
     15  Volume           65 non-null     object
     16  Year             95 non-null     int64
    dtypes: float64(2), int64(2), object(13)
    memory usage: 13.4+ KB

    """

    module_path = dirname(__file__)

    with open(join(module_path, "datasets/auto-trading-selected.rst")) as rst_file:
        fdescr = rst_file.read()

    fdata = pd.read_json(
        join(module_path, "datasets/auto-trading-selected.json"), orient="index"
    )

    return _Bunch(data=fdata, DESCR=fdescr)


def load_test_cleaned():
    """Load and return a test dataset cleaned.

    Args:
        None.

    Returns:
        A dictionary.
    """

    module_path = dirname(__file__)

    with open(join(module_path, "datasets/test-cleaned.rst")) as rst_file:
        fdescr = rst_file.read()

    fdata = pd.read_json(
        join(module_path, "datasets/test-cleaned.json"), orient="records", lines=True,
    )

    return _Bunch(data=fdata, DESCR=fdescr)


def load_test_raw():
    """Load and return a test dataset cleaned.

    Args:
        None.

    Returns:
        A dictionary.
    """

    module_path = dirname(__file__)

    with open(join(module_path, "datasets/test-raw.rst")) as rst_file:
        fdescr = rst_file.read()

    fdata = pd.read_json(
        join(module_path, "datasets/test-raw.json"), orient="records", lines=True
    )

    return _Bunch(data=fdata, DESCR=fdescr)
