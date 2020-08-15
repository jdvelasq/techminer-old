import pandas as pd
import numpy as np

from techminer.params import EXCLUDE_COLS, MULTIVALUED_COLS


##
##
## Term extraction and count
##
##


def _extract_terms(x, column):
    """Extracts unique terms in a column, exploding multvalued columns.

    Args:
        column (str): the column to explode.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame({'Authors': ['xxx', 'xxx; zzz', 'yyy', 'xxx; yyy; zzz']})
    >>> x
             Authors
    0            xxx
    1       xxx; zzz
    2            yyy
    3  xxx; yyy; zzz

    >>> _extract_terms(x, column='Authors')
      Authors
    0     xxx
    1     yyy
    2     zzz

    """
    x = x.copy()
    x[column] = x[column].map(
        lambda w: w.split(";") if not pd.isna(w) and isinstance(w, str) else w
    )
    x = x.explode(column)
    x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
    x = pd.unique(x[column].dropna())
    x = np.sort(x)
    return pd.DataFrame({column: x})


def _count_terms(x, column):
    """Counts the number of different terms in a column.

    Args:
        x (pandas.DataFrame): Biblographic dataframe.
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> x = pd.DataFrame({'Authors': ['xxx', 'xxx; zzz', 'yyy', 'xxx; yyy; zzz']})
    >>> x
             Authors
    0            xxx
    1       xxx; zzz
    2            yyy
    3  xxx; yyy; zzz

    >>> count_terms(x, column='Authors')
    3

    """
    return len(_extract_terms(x, column))


def descriptive_stats(x):
    """
    Descriptive statistics of current dataframe.

    Returns:
        pandas.Series

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          'Authors':  'xxx;xxx, zzz;yyy, xxx, yyy, zzz'.split(','),
    ...          'Authors_ID': '0;1,    3;4;,  4;,  5;,  6;'.split(','),
    ...          'Source_title': ' s0,     s0,   s1,  s1, s2'.split(','),
    ...          'Author_Keywords': 'k0;k1, k0;k2, k3;k2;k1, k4, k5'.split(','),
    ...          'Index_Keywords': 'w0;w1, w0;w2, w3;k1;w1, w4, w5'.split(','),
    ...          'Year': [1990, 1991, 1992, 1993, 1994],
    ...          "Times_Cited": list(range(5)),
    ...          'Num_Authors': [2, 2, 1, 1, 1],
    ...     }
    ... )
    >>> descriptive_stats(x)
                                              value
    Documents                                     5
    Authors                                       3
    Documents per author                       1.67
    Authors per document                        0.6
    Authors of single authored documents          3
    Authors of multi authored documents           2
    Co-authors per document                     1.4
    Authors_ID                                    7
    Source_title                                  3
    Author_Keywords                               6
    Index_Keywords                                7
    Years                                 1990-1994
    Compound annual growth rate               0.0 %
    Times_Cited                                   5
    Average citations per document             2.00
    Num_Authors                                   2
    
    """
    y = {}
    y["Documents"] = str(len(x))
    #
    for column in x.columns:
        if column in EXCLUDE_COLS:
            continue
        if column != "Year":
            y[column] = _count_terms(x, column)
        if column == "Year":
            y["Years"] = str(min(x.Year)) + "-" + str(max(x.Year))
            n = max(x.Year) - min(x.Year) + 1
            Po = len(x.Year[x.Year == min(x.Year)])
            Pn = len(x.Year[x.Year == max(x.Year)])
            cagr = str(round(100 * (np.power(Pn / Po, 1 / n) - 1), 2)) + " %"
            y["Compound annual growth rate"] = cagr
        if column == "Times_Cited":
            y["Average citations per document"] = "{:4.2f}".format(
                x["Times_Cited"].mean()
            )
        if column == "Authors":
            y["Documents per author"] = round(len(x) / _count_terms(x, "Authors"), 2)
            y["Authors per document"] = round(_count_terms(x, "Authors") / len(x), 2)
        if "Num_Authors" in x.columns:
            y["Authors of single authored documents"] = len(x[x["Num_Authors"] == 1])
            y["Authors of multi authored documents"] = len(x[x["Num_Authors"] > 1])
            y["Co-authors per document"] = round(x["Num_Authors"].mean(), 2)
        if "Source_Title" in x.columns:
            y["Average documents per Source title"] = round(
                len(x) / _count_terms(x, "Source_title")
            )
    #
    d = [key for key in y.keys()]
    v = [y[key] for key in y.keys()]
    return pd.DataFrame(v, columns=["value"], index=d)

