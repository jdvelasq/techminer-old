
"""
Analysis by Term per Year
==================================================================================================


 
"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
# from techminer.by_term import citations_by_term, documents_by_term
from techminer.by_year import citations_by_year, documents_by_year
from techminer.explode import __explode
from techminer.keywords import Keywords
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS
from techminer.by_term import get_top_by


def summary_by_term_per_year(x, column, top_by=None, top_n=None, limit_to=None, exclude=None):
    """Computes the number of documents and citations by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited_by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited_by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> summary_by_term_per_year(df, 'Authors')[['Year', 'Authors', 'Cited_by', 'Num_Documents']]
       Year   Authors  Cited_by  Num_Documents
    0  2010  author 0        21              2
    1  2010  author 1        10              1
    2  2010  author 2        10              1
    3  2011  author 1        12              1
    4  2011  author 3        13              1
    5  2012  author 4        14              1
    6  2014  author 4        15              1


    >>> summary_by_term_per_year(df, 'Authors')[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Cited_by']]
       Year   Authors  Perc_Num_Documents  Perc_Cited_by
    0  2010  author 0                50.0          51.22
    1  2010  author 1                25.0          24.39
    2  2010  author 2                25.0          24.39
    3  2011  author 1                50.0          48.00
    4  2011  author 3                50.0          52.00
    5  2012  author 4               100.0         100.00
    6  2014  author 4               100.0         100.00

    >>> terms = ['author 1', 'author 2', 'author 3']
    >>> summary_by_term_per_year(df, 'Authors', limit_to=terms)[['Year', 'Authors', 'Cited_by', 'Num_Documents', 'ID']]
       Year   Authors  Cited_by  Num_Documents   ID
    0  2010  author 1        10              1  [0]
    1  2010  author 2        10              1  [0]
    2  2011  author 1        12              1  [2]
    3  2011  author 3        13              1  [3]

    >>> terms = ['author 1']
    >>> summary_by_term_per_year(df, 'Authors', limit_to=terms)[['Year', 'Authors', 'Cited_by', 'Num_Documents', 'ID']]
       Year   Authors  Cited_by  Num_Documents   ID
    0  2010  author 1        10              1  [0]
    1  2011  author 1        12              1  [2]


    >>> summary_by_term_per_year(df, 'Authors', exclude=terms)[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Cited_by']]
       Year   Authors  Perc_Num_Documents  Perc_Cited_by
    0  2010  author 0                50.0          51.22
    1  2010  author 2                25.0          24.39
    2  2011  author 3                50.0          52.00
    3  2012  author 4               100.0         100.00
    4  2014  author 4               100.0         100.00

    """

    #
    # Computation
    #
    x = x.copy()
    x = __explode(x[["Year", column, "Cited_by", "ID"]], column)
    x["Num_Documents"] = 1
    result = x.groupby([column, "Year"], as_index=False).agg(
        {"Cited_by": np.sum, "Num_Documents": np.size}
    )
    result = result.assign(
        ID=x.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
    )
    result["Cited_by"] = result["Cited_by"].map(lambda x: int(x))

    #
    # Indicators from scientoPy
    #
    groupby = result.groupby('Year').agg({'Num_Documents': np.sum, 'Cited_by':np.sum})
    groupby = groupby.reset_index()
    docs_by_year = {key: value for key, value in zip(groupby.Year, groupby.Num_Documents)}
    cited_by_year = {key: value for key, value in zip(groupby.Year, groupby.Cited_by)}
    result['Documents_by_year'] = result.Year.apply(lambda w: docs_by_year[w])
    result['Documents_by_year'] = result.Documents_by_year.map(lambda w: 1 if w == 0 else w)
    result['Citations_by_year'] = result.Year.apply(lambda w: cited_by_year[w])
    result['Citations_by_year'] = result.Citations_by_year.map(lambda w: 1 if w == 0 else w)
    result = result.assign(Perc_Num_Documents=round(result.Num_Documents / result.Documents_by_year * 100, 2))
    result = result.assign(Perc_Cited_by=round(result.Cited_by / result.Citations_by_year * 100, 2))

    #
    # Filter
    #
    top_terms = get_top_by(x=x, column=column, top_by=top_by, top_n=top_n, limit_to=limit_to, exclude=exclude)

    result = result[result[column].map(lambda w: w in top_terms)]
    result.sort_values(
        ["Year", column, "Num_Documents", "Cited_by"],
        ascending=[True, True, False, False],
        inplace=True,
        ignore_index=True,
    )


    # if isinstance(limit_to, dict):
    #     if column in limit_to.keys():
    #         limit_to = limit_to[column]
    #     else:
    #         limit_to = None

    # if limit_to is not None:
    #     result = result[result[column].map(lambda w: w in limit_to)]

    # if isinstance(exclude, dict):
    #     if column in exclude.keys():
    #         exclude = exclude[column]
    #     else:
    #         exclude = None

    # if exclude is not None:
    #     result = result[result[column].map(lambda w: w not in exclude)]

    # if (top_by == 0 or top_by == "Frequency"):
    #     result.sort_values(
    #         ["Num_Documents", "Cited_by", column],
    #         ascending=[False, False, True],
    #         inplace=True,
    #         ignore_index=True,
    #     )

    # if (top_by == 1 or top_by == "Cited_by"):
    #     result.sort_values(
    #         ["Year", "Cited_by", "Num_Documents", column],
    #         ascending=[True, False, False, True],
    #         inplace=True,
    #         ignore_index=True,
    #     )

    # if top_by is None:
    #     result.sort_values(
    #         [column, "Year", "Cited_by", "Num_Documents"],
    #         ascending=[True, True, False, False],
    #         inplace=True,
    #         ignore_index=True,
    #     )


    # if top_n is not None:
    #     result = result.head(top_n)

    result.reset_index(drop=True)
    return result


def documents_by_term_per_year(x, column, as_matrix=False, top_n=None, limit_to=None, exclude=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited_by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited_by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> documents_by_term_per_year(df, 'Authors')[['Authors', 'Year', 'Num_Documents', 'ID']]
        Authors  Year  Num_Documents      ID
    0  author 0  2010              2  [0, 1]
    1  author 1  2010              1     [0]
    2  author 2  2010              1     [0]
    3  author 1  2011              1     [2]
    4  author 3  2011              1     [3]
    5  author 4  2012              1     [4]
    6  author 4  2014              1     [5]

    >>> documents_by_term_per_year(df, 'Authors')[['Documents_by_year', 'Citations_by_year']]
       Documents_by_year  Citations_by_year
    0                  4                 41
    1                  4                 41
    2                  4                 41
    3                  2                 25
    4                  2                 25
    5                  1                 14
    6                  1                 15

    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         2         1         1         0         0
    2011         0         1         0         1         0
    2012         0         0         0         0         1
    2014         0         0         0         0         1

    >>> terms = ['author 1', 'author 2']
    >>> documents_by_term_per_year(df, 'Authors', limit_to=terms, as_matrix=True)
          author 1  author 2
    2010         1         1
    2011         1         0

    >>> documents_by_term_per_year(df, 'Authors', exclude=terms, as_matrix=True)
          author 0  author 3  author 4
    2010         2         0         0
    2011         0         1         0
    2012         0         0         1
    2014         0         0         1

    """

    result = summary_by_term_per_year(x, column, top_by=0, top_n=top_n, limit_to=limit_to, exclude=exclude)
    
    result.pop("Cited_by")
    result.pop("Perc_Num_Documents")
    result.pop("Perc_Cited_by")
    
    result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Num_Documents", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()

    return result

def perc_documents_by_term_per_year(x, column, as_matrix=False, limit_to=None, exclude=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited_by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited_by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> documents_by_term_per_year(df, 'Authors')
        Authors  Year  Num_Documents      ID  Documents_by_year  Citations_by_year
    0  author 0  2010              2  [0, 1]                  4                 41
    1  author 1  2010              1     [0]                  4                 41
    2  author 2  2010              1     [0]                  4                 41
    3  author 1  2011              1     [2]                  2                 25
    4  author 3  2011              1     [3]                  2                 25
    5  author 4  2012              1     [4]                  1                 14
    6  author 4  2014              1     [5]                  1                 15

    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         2         1         1         0         0
    2011         0         1         0         1         0
    2012         0         0         0         0         1
    2014         0         0         0         0         1

    >>> terms = ['author 1', 'author 2']
    >>> documents_by_term_per_year(df, 'Authors', limit_to=terms, as_matrix=True)
          author 1  author 2
    2010         1         1
    2011         1         0

    >>> documents_by_term_per_year(df, 'Authors', exclude=terms, as_matrix=True)
          author 0  author 3  author 4
    2010         2         0         0
    2011         0         1         0
    2012         0         0         1
    2014         0         0         1

    """

    result = summary_by_term_per_year(x, column, limit_to, exclude)
    result.pop("Cited_by")
    result.pop("Num_Documents")
    result.pop('Perc_Cited_by')
    result.sort_values(
        ["Year", "Perc_Num_Documents", column], ascending=[True, False, True], inplace=True,
    )
    result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Perc_Num_Documents", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    return result


def gant(x, column, limit_to=None, exclude=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2011, 2011, 2012, 2015, 2012, 2016],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 3,author 4,author 4".split(","),
    ...          "Cited_by": list(range(10,17)),
    ...          "ID": list(range(7)),
    ...     }
    ... )
    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         1         1         1         0         0
    2011         1         1         0         0         0
    2012         0         0         0         1         1
    2015         0         0         0         1         0
    2016         0         0         0         0         1

    >>> gant(df, 'Authors')
          author 0  author 1  author 2  author 3  author 4
    2010         1         1         1         0         0
    2011         1         1         0         0         0
    2012         0         0         0         1         1
    2013         0         0         0         1         1
    2014         0         0         0         1         1
    2015         0         0         0         1         1
    2016         0         0         0         0         1


    >>> terms = Keywords(['author 1', 'author 2'])
    >>> gant(df, 'Authors', limit_to=terms)
          author 1  author 2
    2010         1         1
    2011         1         0


    >>> gant(df, 'Authors', exclude=terms)
          author 0  author 3  author 4
    2010         1         0         0
    2011         1         0         0
    2012         0         1         1
    2013         0         1         1
    2014         0         1         1
    2015         0         1         1
    2016         0         0         1

    """


    result = documents_by_term_per_year(
        x, column=column, as_matrix=True, limit_to=limit_to, exclude=exclude
    )
    years = [year for year in range(result.index.min(), result.index.max() + 1)]
    result = result.reindex(years, fill_value=0)
    matrix1 = result.copy()
    matrix1 = matrix1.cumsum()
    matrix1 = matrix1.applymap(lambda x: True if x > 0 else False)
    matrix2 = result.copy()
    matrix2 = matrix2.sort_index(ascending=False)
    matrix2 = matrix2.cumsum()
    matrix2 = matrix2.applymap(lambda x: True if x > 0 else False)
    matrix2 = matrix2.sort_index(ascending=True)
    result = matrix1.eq(matrix2)
    result = result.applymap(lambda x: 1 if x is True else 0)
    return result


def citations_by_term_per_year(x, column, as_matrix=False, top_n=None, limit_to=None, exclude=None):
    """Computes the number of citations by term by year in a column.

    Args:
        column (str): the column to explode.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited_by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited_by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> citations_by_term_per_year(df, 'Authors') [['Year', 'Authors', 'Cited_by', 'ID']]
       Year   Authors  Cited_by      ID
    0  2010  author 0        21  [0, 1]
    1  2010  author 1        10     [0]
    2  2010  author 2        10     [0]
    3  2011  author 1        12     [2]
    4  2011  author 3        13     [3]
    5  2012  author 4        14     [4]
    6  2014  author 4        15     [5]

    >>> citations_by_term_per_year(df, 'Authors')[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Cited_by']]
       Year   Authors  Perc_Num_Documents  Perc_Cited_by
    0  2010  author 0                50.0          51.22
    1  2010  author 1                25.0          24.39
    2  2010  author 2                25.0          24.39
    3  2011  author 1                50.0          48.00
    4  2011  author 3                50.0          52.00
    5  2012  author 4               100.0         100.00
    6  2014  author 4               100.0         100.00

    >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010        21        10        10         0         0
    2011         0        12         0        13         0
    2012         0         0         0         0        14
    2014         0         0         0         0        15

    >>> terms = ['author 1', 'author 2']
    >>> citations_by_term_per_year(df, 'Authors', limit_to=terms)[['Year', 'Authors', 'Cited_by', 'ID']]
       Year   Authors  Cited_by   ID
    0  2010  author 1        10  [0]
    1  2010  author 2        10  [0]
    2  2011  author 1        12  [2]

    >>> citations_by_term_per_year(df, 'Authors', exclude=terms)[['Year', 'Authors', 'Cited_by', 'ID']]
       Year   Authors  Cited_by      ID
    0  2010  author 0        21  [0, 1]
    1  2011  author 3        13     [3]
    2  2012  author 4        14     [4]
    3  2014  author 4        15     [5]

    >>> citations_by_term_per_year(df, 'Authors', limit_to=terms)[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Cited_by']]
       Year   Authors  Perc_Num_Documents  Perc_Cited_by
    0  2010  author 1                25.0          24.39
    1  2010  author 2                25.0          24.39
    2  2011  author 1                50.0          48.00

    >>> citations_by_term_per_year(df, 'Authors', exclude=terms)[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Cited_by']]
       Year   Authors  Perc_Num_Documents  Perc_Cited_by
    0  2010  author 0                50.0          51.22
    1  2011  author 3                50.0          52.00
    2  2012  author 4               100.0         100.00
    3  2014  author 4               100.0         100.00

    """
    result = summary_by_term_per_year(x, column, top_by="Cited_by", top_n=top_n, limit_to=limit_to, exclude=exclude)

    result.pop("Num_Documents")

    result = result.reset_index(drop=True)
    
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Cited_by", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    
    return result

def perc_citations_by_term_per_year(x, column, as_matrix=False, top_n=None, limit_to=None, exclude=None):
    """Computes the number of citations by term by year in a column.

    Args:
        column (str): the column to explode.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited_by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited_by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> citations_by_term_per_year(df, 'Authors')[['Year', 'Authors', 'Cited_by', 'ID']]
       Year   Authors  Cited_by      ID
    0  2010  author 0        21  [0, 1]
    1  2010  author 1        10     [0]
    2  2010  author 2        10     [0]
    3  2011  author 1        12     [2]
    4  2011  author 3        13     [3]
    5  2012  author 4        14     [4]
    6  2014  author 4        15     [5]

    >>> citations_by_term_per_year(df, 'Authors')[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Cited_by']]
       Year   Authors  Perc_Num_Documents  Perc_Cited_by
    0  2010  author 0                50.0          51.22
    1  2010  author 1                25.0          24.39
    2  2010  author 2                25.0          24.39
    3  2011  author 1                50.0          48.00
    4  2011  author 3                50.0          52.00
    5  2012  author 4               100.0         100.00
    6  2014  author 4               100.0         100.00

    >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010        21        10        10         0         0
    2011         0        12         0        13         0
    2012         0         0         0         0        14
    2014         0         0         0         0        15

    >>> terms = ['author 1', 'author 2']
    >>> citations_by_term_per_year(df, 'Authors', limit_to=terms)[['Year', 'Authors', 'Cited_by', 'ID']]
       Year   Authors  Cited_by   ID
    0  2010  author 1        10  [0]
    1  2010  author 2        10  [0]
    2  2011  author 1        12  [2]

    >>> citations_by_term_per_year(df, 'Authors', exclude=terms)[['Year', 'Authors', 'Cited_by', 'ID']]
       Year   Authors  Cited_by      ID
    0  2010  author 0        21  [0, 1]
    1  2011  author 3        13     [3]
    2  2012  author 4        14     [4]
    3  2014  author 4        15     [5]

    """
    result = summary_by_term_per_year(x, column, top_n=top_n, limit_to=limit_to, exclude=exclude)
    result.pop("Num_Documents")
    result.pop("Cited_by")
    result.pop("Perc_Num_Documents")
    result.sort_values(
        ["Year", "Perc_Cited_by", column], ascending=[True, False, False], inplace=True,
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Perc_Cited_by", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    return result


def growth_indicators(x, column, timewindow=2, top_n = None, limit_to=None, exclude=None):
    """Computes the average growth rate of a group of terms.

    Args:
        column (str): the column to explode.
        timewindow (int): time window for analysis
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...   {
    ...     "Year": [2010, 2010, 2011, 2011, 2012, 2013, 2014, 2014],
    ...     "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4,author 0;author 3,author 3;author 4".split(","),
    ...     "Cited_by": list(range(10,18)),
    ...     "ID": list(range(8)),
    ...   }
    ... )
    >>> x
       Year                     Authors  Cited_by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2013                    author 4        15   5
    6  2014           author 0;author 3        16   6
    7  2014           author 3;author 4        17   7

    >>> documents_by_term_per_year(x, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         2         1         1         0         0
    2011         0         1         0         1         0
    2012         0         0         0         0         1
    2013         0         0         0         0         1
    2014         1         0         0         2         1

    >>> growth_indicators(x, 'Authors')
        Authors       AGR  ADY   PDLY  Before 2013  Between 2013-2014
    0  author 3  0.666667  1.0  12.50            1                  2
    1  author 0  0.333333  0.5   6.25            2                  1
    2  author 4  0.000000  1.0  12.50            1                  2

    >>> terms = ['author 3', 'author 4']
    >>> growth_indicators(x, 'Authors', limit_to=terms)
        Authors       AGR  ADY  PDLY  Before 2013  Between 2013-2014
    0  author 3  0.666667  1.0  12.5            1                  2
    1  author 4  0.000000  1.0  12.5            1                  2

    >>> growth_indicators(x, 'Authors', exclude=terms)
        Authors       AGR  ADY  PDLY  Before 2011  Between 2011-2014
    0  author 1 -0.333333  0.5  6.25            1                  1
    1  author 0 -0.333333  0.5  6.25            2                  1

    """

    def compute_agr():
        result = documents_by_term_per_year(
            x, column=column, limit_to=limit_to, exclude=exclude
        )
        years_agr = sorted(set(result.Year))[-(timewindow + 1) :]
        years_agr = [years_agr[0], years_agr[-1]]
        result = result[result.Year.map(lambda w: w in years_agr)]
        result.pop("ID")
        result = pd.pivot_table(
            result,
            columns="Year",
            index=column,
            values="Num_Documents",
            fill_value=0,
        )
        result["AGR"] = 0.0
        result = result.assign(
            AGR=(result[years_agr[1]] - result[years_agr[0]]) / (timewindow + 1)
        )
        result.pop(years_agr[0])
        result.pop(years_agr[1])
        result.columns = list(result.columns)
        result = result.sort_values(by=["AGR", column], ascending=False)
        result.reset_index(drop=True)
        return result

    def compute_ady():
        result = documents_by_term_per_year(
            x, column=column, limit_to=limit_to, exclude=exclude
        )
        years_ady = sorted(set(result.Year))[-timewindow:]
        result = result[result.Year.map(lambda w: w in years_ady)]
        result = result.groupby([column], as_index=False).agg(
            {"Num_Documents": np.sum}
        )
        result = result.rename(columns={"Num_Documents": "ADY"})
        result["ADY"] = result.ADY.map(lambda w: w / timewindow)
        result = result.reset_index(drop=True)
        return result

    def compute_num_documents():
        result = documents_by_term_per_year(
            x, column=column, limit_to=limit_to, exclude=exclude
        )
        years_between = sorted(set(result.Year))[-timewindow:]
        years_before = sorted(set(result.Year))[0:-timewindow]
        between = result[result.Year.map(lambda w: w in years_between)]
        before = result[result.Year.map(lambda w: w in years_before)]
        between = between.groupby([column], as_index=False).agg(
            {"Num_Documents": np.sum}
        )
        between = between.rename(
            columns={
                "Num_Documents": "Between {}-{}".format(
                    years_between[0], years_between[-1]
                )
            }
        )
        before = before.groupby([column], as_index=False).agg(
            {"Num_Documents": np.sum}
        )
        before = before.rename(
            columns={"Num_Documents": "Before {}".format(years_between[0])}
        )
        result = pd.merge(before, between, on=column)
        return result

    result = compute_agr()
    ady = compute_ady()
    result = pd.merge(result, ady, on=column)
    result = result.assign(PDLY=round(result.ADY / len(x) * 100, 2))
    num_docs = compute_num_documents()
    result = pd.merge(result, num_docs, on=column)
    result = result.reset_index(drop=True)
    return result



##
##
## APP
##
##

WIDGET_WIDTH = "180px"
LEFT_PANEL_HEIGHT = "655px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "720px", 0]

##
##
##  Time Analysis
##
##
def __APP0__(x, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
    controls = [
        # 0
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "analysis_type",
            "desc": "Analysis type:",
            "widget": widgets.Dropdown(
                options=["Frequency", "Citation", "Perc. Frequency", "Perc. Cited by"],
                value="Frequency",
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "plot_type",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Heatmap", "Bubble plot", "Gant diagram", 'Lines plot', "Table"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                    options=list(range(5, 51, 5)),
                    ensure_option=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 5
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
                ),
        },
        # 6
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
                ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        term = kwargs["term"]
        analysis_type = kwargs["analysis_type"]
        plot_type = kwargs["plot_type"]
        cmap = kwargs["cmap"]
        top_n = kwargs["top_n"]
        figsize_width = int(kwargs['figsize_width'])
        figsize_height = int(kwargs['figsize_height'])
        #
        plots = {"Heatmap": plt.heatmap, "Gant diagram": plt.gant, "Bubble plot": plt.bubble, "Lines": plt.plot, "Table":None}
        plot = plots[plot_type]
        #
        if analysis_type == "Frequency":
            matrix = documents_by_term_per_year(x, term, as_matrix=True, top_n=top_n, limit_to=limit_to, exclude=exclude)
        
        if analysis_type == 'Citation':
            matrix = citations_by_term_per_year(x, term, as_matrix=True, topn=top_n, limit_to=limit_to, exclude=exclude)

        if analysis_type == 'Perc. Frequency':
            matrix = perc_documents_by_term_per_year(x, term, as_matrix=True, top_n=top_n, limit_to=limit_to, exclude=exclude)

        if analysis_type == 'Perc. Cited by':
            matrix = perc_citations_by_term_per_year(x, term, as_matrix=True, top_n=top_n, limit_to=limit_to, exclude=exclude)


        output.clear_output()
        with output:
            if plot_type == "Heatmap":
                display(plot(matrix, cmap=cmap, figsize=(figsize_width, figsize_height)))
            if plot_type == "Gant diagram":
                display(plot(matrix, cmap=cmap, figsize=(figsize_width, figsize_height)))
            if plot_type == "Bubble plot":
                display(plot(matrix.transpose(), axis=0, cmap=cmap, figsize=(figsize_width, figsize_height)))
            if plot_type == "Lines plot":
                display(plot(matrix, cmap=cmap, figsize=(figsize_width, figsize_height)))
            if plot_type == 'Table':
                with pd.option_context("display.max_columns", 50):
                    display(matrix)


    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                    if control["desc"] not in ["Figsize"]
                ] + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([
                        controls[-2]["widget"],
                        controls[-1]["widget"],
                    ])
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
        ]
    )


##
##
##  Growth Indicators
##
##
def __APP1__(x, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
    controls = [
        # 0
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                    options=[z for z in COLUMNS if z in x.columns],
                    ensure_option=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 1
        {
            "arg": "analysis_type",
            "desc": "Analysis type:",
            "widget": widgets.Dropdown(
                    options=[
                        "Average Growth Rate",
                        "Average Documents per Year",
                        "Percentage of Documents in Last Years",
                        "Number of Document Published",
                    ],
                    value="Average Growth Rate",
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 2
        {
            "arg": "time_window",
            "desc": "Time window:",
            "widget": widgets.Dropdown(
                    options=["2", "3", "4", "5"],
                    value="2",
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 3
        {
            "arg": "plot_type",
            "desc": "Plot type:",
            "widget": widgets.Dropdown(
                    options=["bar", "barh"],
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 4
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                    options=list(range(5, 51, 5)),
                    ensure_option=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 6
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    layout=Layout(width="88px"),
                ),
        },
        # 7
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    layout=Layout(width="88px"),
                ),
        },

    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        term = kwargs['term']
        cmap = kwargs['cmap']
        analysis_type = kwargs['analysis_type']
        top_n = kwargs['top_n']
        plot_type = kwargs['plot_type']
        time_window = int(kwargs['time_window'])
        figsize_width = int(kwargs['figsize_width'])
        figsize_height = int(kwargs['figsize_height'])        
        #
        plots = {"bar": plt.bar, "barh": plt.barh}
        plot = plots[plot_type]
        #
        df = growth_indicators(x, term, timewindow=time_window, limit_to=limit_to, exclude=exclude)
        output.clear_output()

        with output:
            if analysis_type == "Average Growth Rate":
                df = df.sort_values('AGR', ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                display(plot(df[[term, 'AGR']], cmap=cmap, figsize=(figsize_width, figsize_height)))
            if analysis_type == "Average Documents per Year":
                df = df.sort_values('ADY', ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                display(plot(df[[term, 'ADY']], cmap=cmap, figsize=(figsize_width, figsize_height)))
            if analysis_type == "Percentage of Documents in Last Years":
                df = df.sort_values('PDLY', ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                display(plot(df[[term, 'PDLY']], cmap=cmap, figsize=(figsize_width, figsize_height)))
            if analysis_type == "Number of Document Published":
                df['Num_Documents'] = df[df.columns[-2]] + df[df.columns[-1]]
                df = df.sort_values('Num_Documents', ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                df.pop('Num_Documents')
                if plot_type == 'bar':
                    display(plt.stacked_bar(df[[term, df.columns[-2], df.columns[-1]]], figsize=(figsize_width, figsize_height), cmap=cmap))
                if plot_type == 'barh':
                    display(plt.stacked_barh(df[[term, df.columns[-2], df.columns[-1]]], figsize=(figsize_width, figsize_height), cmap=cmap))


    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                    if control["desc"] not in ["Figsize"]
                ] + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([
                        controls[-2]["widget"],
                        controls[-1]["widget"],
                    ])
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
        ]
    )



def app(df, limit_to=None, exclude=None):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__APP0__(df, limit_to, exclude), __APP1__(df, limit_to, exclude)]
    body.set_title(0, "Time Analysis")
    body.set_title(1, "Growth Indicators")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Summary by Term per Year"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )



#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
