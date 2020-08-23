import pandas as pd
import numpy as np

from techminer.core.params import EXCLUDE_COLS, MULTIVALUED_COLS


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


def descriptive_stats(input_file="techminer.csv"):

    x = pd.read_csv(input_file)

    y = {}
    y["Documents"] = str(len(x))
    #
    for column in sorted(x.columns):

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
            y["Authors of single-authored documents"] = len(x[x["Num_Authors"] == 1])
            y["Authors of multi-authored documents"] = len(x[x["Num_Authors"] > 1])
            y["Co-authors per document"] = round(x["Num_Authors"].mean(), 2)

        if "Source_Title" in x.columns:
            y["Average documents per Source title"] = round(
                len(x) / _count_terms(x, "Source_title")
            )

        if "Global_References" in x.columns:
            y["Average global references per document"] = round(
                _count_terms(x, "Global_References") / len(x)
            )
    #
    d = [key for key in sorted(y.keys())]
    v = [y[key] for key in sorted(y.keys())]
    return pd.DataFrame(v, columns=["value"], index=d)

