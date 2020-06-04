
"""
Descriptive Statistics
==================================================================================================



"""
import pandas as pd
import numpy as np
from techminer.explode import MULTIVALUED_COLS

##
##
## Term extraction and count
##
##



def extract_terms(x, column):
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

    >>> extract_terms(x, column='Authors')
      Authors
    0     xxx
    1     yyy
    2     zzz

    """
    if column in MULTIVALUED_COLS:
        x = x.copy()
        x[column] = x[column].map(lambda w: w.split(";") if not pd.isna(w) else w)
        x = x.explode(column)
        x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
    else:
        x = x[[column]].copy()
    x = pd.unique(x[column].dropna())
    x = np.sort(x)
    return pd.DataFrame({column: x})


def count_terms(x, column):
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
    return len(extract_terms(x, column))


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
    ...          'Author(s) ID': '0;1,    3;4;,  4;,  5;,  6;'.split(','),
    ...          'Source title': ' s0,     s0,   s1,  s1, s2'.split(','),
    ...          'Author Keywords': 'k0;k1, k0;k2, k3;k2;k1, k4, k5'.split(','),
    ...          'Index Keywords': 'w0;w1, w0;w2, w3;k1;w1, w4, w5'.split(','),
    ...          'Year': [1990, 1991, 1992, 1993, 1994],
    ...          'Cited by': list(range(5)),
    ...          'Num Authors': [2, 2, 1, 1, 1],
    ...     }
    ... )
    >>> descriptive_stats(x)
                                             value
    Articles                                     5
    Years                                1990-1994
    Average citations per article             2.00
    Authors                                      3
    Author(s) ID                                 7
    Articles per author                       1.67
    Authors per article                        0.6
    Author Keywords                              6
    Index Keywords                               7
    Source titles                                5
    Authors of single authored articles          3
    Authors of multi authored articles           2
    Co-authors per article                     1.4
    Average articles per Source title            1
    Compound annual growth rate              0.0 %

    """
    descriptions = [
        "Articles",
        "Years",
        "Average citations per article",
        "Authors",
        "Author(s) ID",
        "Authors of single authored articles",
        "Authors of multi authored articles",
        "Articles per author",
        "Authors per article",
        "Co-authors per article",
        "Author Keywords",
        "Index Keywords",
        "Source titles",
        "Average articles per Source title",
        "Compound Annual Growth Rate",
    ]
    y = {}
    y["Articles"] = str(len(x))
    y["Years"] = str(min(x.Year)) + "-" + str(max(x.Year))
    y["Average citations per article"] = "{:4.2f}".format(x["Cited by"].mean())
    y["Authors"] = count_terms(x, "Authors")
    y["Author(s) ID"] = count_terms(x, "Author(s) ID")
    y["Articles per author"] = round(len(x) / count_terms(x, "Authors"), 2)
    y["Authors per article"] = round(count_terms(x, "Authors") / len(x), 2)
    y["Author Keywords"] = count_terms(x, "Author Keywords")
    y["Index Keywords"] = count_terms(x, "Index Keywords")
    y["Source titles"] = count_terms(x, "Source title")

    y["Authors of single authored articles"] = len(x[x["Num Authors"] == 1])
    y["Authors of multi authored articles"] = len(x[x["Num Authors"] > 1])
    y["Co-authors per article"] = round(x["Num Authors"].mean(), 2)
    y["Average articles per Source title"] = round(
        len(x) / count_terms(x, "Source title")
    )
    n = max(x.Year) - min(x.Year) + 1
    Po = len(x.Year[x.Year == min(x.Year)])
    Pn = len(x.Year[x.Year == max(x.Year)])
    cagr = str(round(100 * (np.power(Pn / Po, 1 / n) - 1), 2)) + " %"
    y["Compound annual growth rate"] = cagr
    #
    d = [key for key in y.keys()]
    v = [y[key] for key in y.keys()]
    return pd.DataFrame(v, columns=["value"], index=d)



#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()