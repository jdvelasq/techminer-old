"""
Analytics
==================================================================================================


"""

import pandas as pd
import numpy as np
from techminer.text import remove_accents, extract_country, extract_institution
from techminer.keywords import Keywords

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NORMALIZED_NAMES = {
    "J9": "Abbreviated Source Title",
    "AB": "Abstract",
    "OA": "Access Type",
    "C1": "Affiliations",
    "AR": "Art. No.",
    "DE": "Author Keywords",
    "AU_CO": "Countries",
    "RI": "Author(s) ID",
    "AU_IN": "Institutions",
    "AU_C1": "Authors with affiliations",
    "AU": "Authors",
    "TC": "Cited by",
    "CR": "Cited references",
    "DOI": "DI",
    "DT": "Document type",
    "BE": "Editors",
    "UT": "EID",
    "ID": "Index Keywords",
    "BN": "ISBN",
    "SN": "ISSN",
    "IS": "Issue",
    "DE_ID": "Keywords",
    "LA": "Language of Original Document",
    "PG": "Page count",
    "EP": "Page end",
    "BP": "Page start",
    "PU": "Publisher",
    "PM": "PubMed ID",
    "SO": "Source title",
    "FN": "Source",
    "SC": "Subject",
    "TI": "Title",
    "VL": "Volume",
    "PT": "Year",
}

MULTIVALUED_COLS = [
    "Affiliations",
    "Author Keywords",
    "Author(s) Country",
    "Author(s) ID",
    "Author(s) Institution",
    "Authors with affiliations",
    "Authors",
    "Countries",
    "Index Keywords",
    "Institutions",
    "Keywords",
]

##
##
## Auxiliary Functions
##
##
def __explode(x, column):
    """Transform each element of a field to a row, reseting index values.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.

    Returns:
        DataFrame. Exploded dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...         "Authors": "author 0;author 1;author 2,author 3,author 4".split(","),
    ...         "ID": list(range(3)),
    ...      }
    ... )
    >>> x
                          Authors  ID
    0  author 0;author 1;author 2   0
    1                    author 3   1
    2                    author 4   2

    >>> __explode(x, 'Authors')
        Authors  ID
    0  author 0   0
    1  author 1   0
    2  author 2   0
    3  author 3   1
    4  author 4   2

    """
    if column in MULTIVALUED_COLS:
        x = x.copy()
        x[column] = x[column].map(
            lambda w: sorted(list(set(w.split(";")))) if isinstance(w, str) else w
        )
        x = x.explode(column)
        x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
        x = x.reset_index(drop=True)
    return x


##
##
##  Data importation
##
##
def load_scopus(x):
    """Import filter for Scopus data.
    """
    #
    logging.info("Renaming and selecting columns ...")
    x = x.copy()
    x = x.rename(columns=NORMALIZED_NAMES)
    x = x[[w for w in x.columns if w in NORMALIZED_NAMES.values()]]
    #
    logging.info("Formatting author names ...")
    x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)
    if "Authors" in x.columns:
        x["Authors"] = x.Authors.map(
            lambda w: w.replace(",", ";").replace(".", "") if pd.isna(w) is False else w
        )
    #
    logging.info("Removing part of titles in foreing languages ...")
    if "Title" in x.columns:
        x["Title"] = x.Title.map(
            lambda w: w[0 : w.find("[")] if pd.isna(w) is False and w[-1] == "]" else w
        )
    #
    logging.info("Fusioning author and index keywords ...")
    if (
        "Author Keywords" in x.columns.tolist()
        and "Index Keywords" in x.columns.tolist()
    ):
        author_keywords = x["Author Keywords"].map(
            lambda x: x.split(";") if pd.isna(x) is False else []
        )
        index_keywords = x["Index Keywords"].map(
            lambda x: x.split(";") if pd.isna(x) is False else []
        )
        keywords = author_keywords + index_keywords
        keywords = keywords.map(lambda w: [e for e in w if e != ""])
        keywords = keywords.map(lambda w: [e.strip() for e in w])
        keywords = keywords.map(lambda w: sorted(set(w)))
        keywords = keywords.map(lambda w: ";".join(w))
        keywords = keywords.map(lambda w: None if w == "" else w)
        x["Keywords"] = keywords
    #
    logging.info("Extracting countries from affiliations ...")
    if "Affiliations" in x.columns:

        x["Countries"] = x.Affiliations.map(
            lambda w: extract_country(w) if pd.isna(w) is False else w
        )
    #
    logging.info("Extracting institutions from affiliations ...")
    if "Affiliations" in x.columns:
        x["Institutions"] = x.Affiliations.map(
            lambda w: extract_institution(w) if pd.isna(w) is False else w
        )
    #
    logging.info("Extracting country of 1st author ...")
    if "Countries" in x.columns:
        x["Country 1st"] = x["Countries"].map(
            lambda w: w.split(";")[0] if not pd.isna(w) else w
        )
    #
    logging.info("Extracting affiliation of 1st author ...")
    if "Institutions" in x.columns:
        x["Institution 1st"] = x["Institutions"].map(
            lambda w: w.split(";")[0] if not pd.isna(w) else w
        )
    #
    logging.info("Counting number of authors ...")
    x["Num Authors"] = x["Authors"].map(
        lambda w: len(w.split(";")) if not pd.isna(w) else 0
    )
    #
    x["ID"] = range(len(x))
    #
    return x


##
##
##  Data coverage
##
##


def coverage(x):
    """Reports the number of not `None` elements for column in a dataframe.

    Returns:
        Pandas DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------


    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...   {
    ...      "Authors": "author 0,author 0,author 0;author 0;author 0;author 1;author 2;author 0".split(";"),
    ...      "Author(s) ID": "0;1;2;,3;,4;,5;,6;,7:".split(','),
    ...      "ID": list(range(6)),
    ...      "Source title": ['source {:d}'.format(w) for w in range(5)] + [pd.NA],
    ...      "None field": [pd.NA] * 6,
    ...   }
    ... )
    >>> x
                          Authors Author(s) ID  ID Source title None field
    0  author 0,author 0,author 0       0;1;2;   0     source 0       <NA>
    1                    author 0           3;   1     source 1       <NA>
    2                    author 0           4;   2     source 2       <NA>
    3                    author 1           5;   3     source 3       <NA>
    4                    author 2           6;   4     source 4       <NA>
    5                    author 0           7:   5         <NA>       <NA>

    >>> coverage(x)
             Column  Number of items Coverage (%)
    0       Authors                6      100.00%
    1  Author(s) ID                6      100.00%
    2            ID                6      100.00%
    3  Source title                5       83.33%
    4    None field                0        0.00%


    """

    return pd.DataFrame(
        {
            "Column": x.columns,
            "Number of items": [len(x) - x[col].isnull().sum() for col in x.columns],
            "Coverage (%)": [
                "{:5.2%}".format((len(x) - x[col].isnull().sum()) / len(x))
                for col in x.columns
            ],
        }
    )


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
    cagr = str(round(100 * (np.power(Pn / Po, n) - 1), 2)) + " %"
    y["Compound annual growth rate"] = cagr
    #
    d = [key for key in y.keys()]
    v = [y[key] for key in y.keys()]
    return pd.DataFrame(v, columns=["value"], index=d)


def summary_by_year(df):
    """Computes the number of document and the number of total citations per year.
    This funciton adds the missing years in the sequence.


    Args:
        df (pandas.DataFrame): bibliographic dataframe.


    Returns:
        pandas.DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Cited by  ID
    0  2010        10   0
    1  2010        11   1
    2  2011        12   2
    3  2011        13   3
    4  2012        14   4
    5  2016        15   5

    >>> summary_by_year(df)[['Year', 'Cited by', 'Num Documents', 'ID']]
       Year  Cited by  Num Documents      ID
    0  2010        21              2  [0, 1]
    1  2011        25              2  [2, 3]
    2  2012        14              1     [4]
    3  2013         0              0      []
    4  2014         0              0      []
    5  2015         0              0      []
    6  2016        15              1     [5]

    >>> summary_by_year(df)[['Num Documents (Cum)', 'Cited by (Cum)', 'Avg. Cited by']]
       Num Documents (Cum)  Cited by (Cum)  Avg. Cited by
    0                    2              21           10.5
    1                    4              46           12.5
    2                    5              60           14.0
    3                    5              60            0.0
    4                    5              60            0.0
    5                    5              60            0.0
    6                    6              75           15.0

    """
    data = df[["Year", "Cited by", "ID"]].explode("Year")
    data["Num Documents"] = 1
    result = data.groupby("Year", as_index=False).agg(
        {"Cited by": np.sum, "Num Documents": np.size}
    )
    result = result.assign(
        ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
    )
    result["Cited by"] = result["Cited by"].map(lambda x: int(x))
    years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
    result = result.set_index("Year")
    result = result.reindex(years, fill_value=0)
    result["ID"] = result["ID"].map(lambda x: [] if x == 0 else x)
    result.sort_values(
        "Year", ascending=True, inplace=True,
    )
    result["Num Documents (Cum)"] = result["Num Documents"].cumsum()
    result["Cited by (Cum)"] = result["Cited by"].cumsum()
    result["Avg. Cited by"] = result["Cited by"] / result["Num Documents"]
    result["Avg. Cited by"] = result["Avg. Cited by"].map(
        lambda x: 0 if pd.isna(x) else x
    )
    result = result.reset_index()
    return result


# def documents_by_year(x, cumulative=False):
#     """Computes the number of documents per year.
#     This function adds the missing years in the sequence.

#     Args:
#         cumulative (bool): cumulate values per year.

#     Returns:
#         DataFrame.

#     """
#     result = summary_by_year(x, cumulative)
#     result.pop("Cited by")
#     result = result.reset_index(drop=True)
#     return result


# def citations_by_year(x, cumulative=False):
#     """Computes the number of citations by year.
#     This function adds the missing years in the sequence.

#     Args:
#         cumulative (bool): cumulate values per year.

#     Returns:
#         DataFrame.

#     """
#     result = summary_by_year(x, cumulative)
#     result.pop("Num Documents")
#     result = result.reset_index(drop=True)
#     return result


##
##
##  Analysis by term
##
##


def summary_by_term(x, column, keywords=None):
    """Summarize the number of documents and citations by term in a dataframe.

    Args:
        column (str): the column to explode.
        keywords (int, list): filter the results.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> summary_by_term(x, 'Authors')
        Authors  Num Documents  Cited by      ID
    0  author 0              2        21  [0, 1]
    1  author 1              2        22  [0, 2]
    2  author 2              1        10     [0]
    3  author 3              1        13     [3]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> summary_by_term(x, 'Authors', keywords=keywords)
        Authors  Num Documents  Cited by      ID
    0  author 1              2        22  [0, 2]
    1  author 2              1        10     [0]

    """
    x = x.copy()
    x = __explode(x[[column, "Cited by", "ID"]], column)
    x["Num Documents"] = 1
    result = x.groupby(column, as_index=False).agg(
        {"Num Documents": np.size, "Cited by": np.sum}
    )
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result["Cited by"] = result["Cited by"].map(lambda x: int(x))
    if keywords is not None:
        result = result[result[column].map(lambda w: w in keywords)]
    result.sort_values(
        [column, "Num Documents", "Cited by"],
        ascending=[True, False, False],
        inplace=True,
        ignore_index=True,
    )
    return result


def documents_by_term(x, column, keywords=None):
    """Computes the number of documents per term in a given column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> documents_by_term(x, 'Authors')
        Authors  Num Documents      ID
    0  author 0              2  [0, 1]
    1  author 1              2  [0, 2]
    2  author 2              1     [0]
    3  author 3              1     [3]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> documents_by_term(x, 'Authors', keywords=keywords)
        Authors  Num Documents      ID
    0  author 1              2  [0, 2]
    1  author 2              1     [0]

    """

    result = summary_by_term(x, column, keywords)
    result.pop("Cited by")
    result.sort_values(
        ["Num Documents", column],
        ascending=[False, True],
        inplace=True,
        ignore_index=True,
    )
    return result


def citations_by_term(x, column, keywords=None):
    """Computes the number of citations by item in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> citations_by_term(x, 'Authors')
        Authors  Cited by      ID
    0  author 1        22  [0, 2]
    1  author 0        21  [0, 1]
    2  author 3        13     [3]
    3  author 2        10     [0]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> citations_by_term(x, 'Authors', keywords=keywords)
        Authors  Cited by      ID
    0  author 1        22  [0, 2]
    1  author 2        10     [0]


    """
    result = summary_by_term(x, column, keywords)
    result.pop("Num Documents")
    result.sort_values(
        ["Cited by", column], ascending=[False, True], inplace=True, ignore_index=True,
    )
    return result


def summary_by_term_per_year(x, column, keywords=None):
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
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> summary_by_term_per_year(df, 'Authors')
        Authors  Year  Cited by  Num Documents      ID
    0  author 0  2010        21              2  [0, 1]
    1  author 1  2010        10              1     [0]
    2  author 2  2010        10              1     [0]
    3  author 1  2011        12              1     [2]
    4  author 3  2011        13              1     [3]
    5  author 4  2012        14              1     [4]
    6  author 4  2014        15              1     [5]

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> summary_by_term_per_year(df, 'Authors', keywords=keywords)
        Authors  Year  Cited by  Num Documents   ID
    0  author 1  2010        10              1  [0]
    1  author 2  2010        10              1  [0]
    2  author 1  2011        12              1  [2]
    3  author 3  2011        13              1  [3]

    """
    data = __explode(x[["Year", column, "Cited by", "ID"]], column)
    data["Num Documents"] = 1
    result = data.groupby([column, "Year"], as_index=False).agg(
        {"Cited by": np.sum, "Num Documents": np.size}
    )
    result = result.assign(
        ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
    )
    result["Cited by"] = result["Cited by"].map(lambda x: int(x))
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[column].map(lambda w: w in keywords)]
    result.sort_values(
        ["Year", column], ascending=True, inplace=True, ignore_index=True,
    )
    return result


def documents_by_term_per_year(x, column, as_matrix=False, minmax=None, keywords=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
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
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> documents_by_term_per_year(df, 'Authors')
        Authors  Year  Num Documents      ID
    0  author 0  2010              2  [0, 1]
    1  author 1  2010              1     [0]
    2  author 2  2010              1     [0]
    3  author 1  2011              1     [2]
    4  author 3  2011              1     [3]
    5  author 4  2012              1     [4]
    6  author 4  2014              1     [5]

    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         2         1         1         0         0
    2011         0         1         0         1         0
    2012         0         0         0         0         1
    2014         0         0         0         0         1

    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True, minmax=(2, None))
          author 0
    2010         2

    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True, minmax=(0, 1))
          author 1  author 2  author 3  author 4
    2010         1         1         0         0
    2011         1         0         1         0
    2012         0         0         0         1
    2014         0         0         0         1

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> documents_by_term_per_year(df, 'Authors', keywords=keywords, as_matrix=True)
          author 1  author 2  author 3
    2010         1         1         0
    2011         1         0         1

    """

    result = summary_by_term_per_year(x, column, keywords)
    result.pop("Cited by")
    if minmax is not None:
        min_value, max_value = minmax
        if min_value is not None:
            result = result[result["Num Documents"] >= min_value]
        if max_value is not None:
            result = result[result["Num Documents"] <= max_value]
    result.sort_values(
        ["Year", "Num Documents", column], ascending=[True, False, True], inplace=True,
    )
    result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Num Documents", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    return result


def gant(x, column, minmax=None, keywords=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
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
    ...          "Cited by": list(range(10,17)),
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

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> gant(df, 'Authors', keywords=keywords)
          author 1  author 2  author 3
    2010         1         1         0
    2011         1         0         0
    2012         0         0         1
    2013         0         0         1
    2014         0         0         1
    2015         0         0         1

    """
    result = documents_by_term_per_year(
        x, column=column, as_matrix=True, minmax=minmax, keywords=keywords
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


def citations_by_term_per_year(x, column, as_matrix=False, minmax=None, keywords=None):
    """Computes the number of citations by term by year in a column.

    Args:
        column (str): the column to explode.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
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
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> citations_by_term_per_year(df, 'Authors')
        Authors  Year  Cited by      ID
    0  author 0  2010        21  [0, 1]
    1  author 2  2010        10     [0]
    2  author 1  2010        10     [0]
    3  author 3  2011        13     [3]
    4  author 1  2011        12     [2]
    5  author 4  2012        14     [4]
    6  author 4  2014        15     [5]

    >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010        21        10        10         0         0
    2011         0        12         0        13         0
    2012         0         0         0         0        14
    2014         0         0         0         0        15

    >>> citations_by_term_per_year(df, 'Authors', as_matrix=True, minmax=(12, 15))
          author 1  author 3  author 4
    2011        12        13         0
    2012         0         0        14
    2014         0         0        15

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> citations_by_term_per_year(df, 'Authors', keywords=keywords)
        Authors  Year  Cited by   ID
    0  author 2  2010        10  [0]
    1  author 1  2010        10  [0]
    2  author 3  2011        13  [3]
    3  author 1  2011        12  [2]

    """
    result = summary_by_term_per_year(x, column, keywords)
    result.pop("Num Documents")
    if minmax is not None:
        min_value, max_value = minmax
        if min_value is not None:
            result = result[result["Cited by"] >= min_value]
        if max_value is not None:
            result = result[result["Cited by"] <= max_value]
    result.sort_values(
        ["Year", "Cited by", column], ascending=[True, False, False], inplace=True,
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Cited by", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    return result


def summary_occurrence(x, column, keywords=None):
    """Summarize occurrence and citations by terms in a column of a dataframe.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors  Cited by  ID
    0       A         0   0
    1       A         1   1
    2     A;B         2   2
    3       B         3   3
    4   A;B;C         4   4
    5       D         5   5
    6     B;D         6   6

    >>> summary_occurrence(df, column='Authors')
       Authors (IDX) Authors (COL)  Num Documents  Cited by            ID
    0              A             A              4         7  [0, 1, 2, 4]
    1              A             B              2         6        [2, 4]
    2              A             C              1         4           [4]
    3              B             A              2         6        [2, 4]
    4              B             B              4        15  [2, 3, 4, 6]
    5              B             C              1         4           [4]
    6              B             D              1         6           [6]
    7              C             A              1         4           [4]
    8              C             B              1         4           [4]
    9              C             C              1         4           [4]
    10             D             B              1         6           [6]
    11             D             D              2        11        [5, 6]

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> summary_occurrence(df, 'Authors', keywords=keywords)
      Authors (IDX) Authors (COL)  Num Documents  Cited by            ID
    0             A             A              4         7  [0, 1, 2, 4]
    1             A             B              2         6        [2, 4]
    2             B             A              2         6        [2, 4]
    3             B             B              4        15  [2, 3, 4, 6]


    """

    def generate_pairs(w):
        w = [x.strip() for x in w.split(sep)]
        result = []
        for idx0 in range(len(w)):
            for idx1 in range(len(w)):
                result.append((w[idx0], w[idx1]))
        return result

    sep = ";" if column in MULTIVALUED_COLS else None

    data = x.copy()
    data = data[[column, "Cited by", "ID"]]
    data = data.dropna()
    data["count"] = 1
    data["pairs"] = data[column].map(lambda x: generate_pairs(x))
    data = data[["pairs", "count", "Cited by", "ID"]]
    data = data.explode("pairs")
    result = (
        data.groupby("pairs", as_index=False)
        .agg({"Cited by": np.sum, "count": np.sum, "ID": list})
        .rename(columns={"count": "Num Documents"})
    )
    result["Cited by"] = result["Cited by"].map(int)
    result[column + " (IDX)"] = result["pairs"].map(lambda x: x[0])
    result[column + " (COL)"] = result["pairs"].map(lambda x: x[1])
    result.pop("pairs")
    result = result[
        [column + " (IDX)", column + " (COL)", "Num Documents", "Cited by", "ID"]
    ]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[column + " (IDX)"].map(lambda w: w in keywords)]
        result = result[result[column + " (COL)"].map(lambda w: w in keywords)]
    result = result.sort_values(
        [column + " (IDX)", column + " (COL)"], ignore_index=True,
    )
    return result


def occurrence(x, column, as_matrix=False, minmax=None, keywords=None, retmaxval=False):
    """Computes the occurrence between the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (list, Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors  Cited by  ID
    0       A         0   0
    1       A         1   1
    2     A;B         2   2
    3       B         3   3
    4   A;B;C         4   4
    5       D         5   5
    6     B;D         6   6

    >>> occurrence(df, column='Authors')
       Authors (IDX) Authors (COL)  Num Documents            ID
    0              A             A              4  [0, 1, 2, 4]
    1              B             B              4  [2, 3, 4, 6]
    2              A             B              2        [2, 4]
    3              B             A              2        [2, 4]
    4              D             D              2        [5, 6]
    5              A             C              1           [4]
    6              B             C              1           [4]
    7              B             D              1           [6]
    8              C             A              1           [4]
    9              C             B              1           [4]
    10             C             C              1           [4]
    11             D             B              1           [6]

    >>> occurrence(df, column='Authors', as_matrix=True)
       A  B  C  D
    A  4  2  1  0
    B  2  4  1  1
    C  1  1  1  0
    D  0  1  0  2

    >>> occurrence(df, column='Authors', as_matrix=True, minmax=(2,3))
       A  B  D
    A  0  2  0
    B  2  0  0
    D  0  0  2

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> occurrence(df, 'Authors', as_matrix=True, keywords=keywords)
       A  B
    A  4  2
    B  2  4

    """

    def generate_dic(column):
        new_names = documents_by_term(x, column)
        new_names = {
            term: "{:s} [{:d}]".format(term, docs_per_term)
            for term, docs_per_term in zip(
                new_names[column], new_names["Num Documents"],
            )
        }
        return new_names

    column_IDX = column + " (IDX)"
    column_COL = column + " (COL)"

    result = summary_occurrence(x, column)
    result.pop("Cited by")
    maxval = result["Num Documents"].max()
    if minmax is not None:
        min_value, max_value = minmax
        if min_value is not None:
            if min_value > maxval:
                min_value = maxval
            result = result[result["Num Documents"] >= min_value]
        if max_value is not None:
            if max_value > maxval:
                max_value = maxval
            result = result[result["Num Documents"] <= max_value]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[column_IDX].map(lambda w: w in keywords)]
        result = result[result[column_COL].map(lambda w: w in keywords)]
    result.sort_values(
        ["Num Documents", column_IDX, column_COL],
        ascending=[False, True, True],
        inplace=True,
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result,
            values="Num Documents",
            index=column_IDX,
            columns=column_COL,
            fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    if retmaxval is True:
        return result, maxval
    return result


def self_citation(
    x, column, as_matrix=False, minmax=None, keywords=None, retmaxval=False
):
    """Computes the cocitation between the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (list, Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors  Cited by  ID
    0       A         0   0
    1       A         1   1
    2     A;B         2   2
    3       B         3   3
    4   A;B;C         4   4
    5       D         5   5
    6     B;D         6   6

    >>> self_citation(df, column='Authors')
       Authors (IDX) Authors (COL)  Cited by            ID
    0              B             B        15  [2, 3, 4, 6]
    1              D             D        11        [5, 6]
    2              A             A         7  [0, 1, 2, 4]
    3              A             B         6        [2, 4]
    4              B             A         6        [2, 4]
    5              B             D         6           [6]
    6              D             B         6           [6]
    7              A             C         4           [4]
    8              B             C         4           [4]
    9              C             A         4           [4]
    10             C             B         4           [4]
    11             C             C         4           [4]

    >>> self_citation(df, column='Authors', as_matrix=True)
       A   B  C   D
    A  7   6  4   0
    B  6  15  4   6
    C  4   4  4   0
    D  0   6  0  11

    >>> self_citation(df, column='Authors', as_matrix=True, minmax=(6,15))
       A   B   D
    A  7   6   0
    B  6  15   6
    D  0   6  11

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> self_citation(df, 'Authors', as_matrix=True, keywords=keywords)
       A   B
    A  7   6
    B  6  15


    """

    def generate_dic(column):
        new_names = citations_by_term(x, column)
        new_names = {
            term: "{:s} [{:d}]".format(term, docs_per_term)
            for term, docs_per_term in zip(new_names[column], new_names["Cited by"],)
        }
        return new_names

    column_IDX = column + " (IDX)"
    column_COL = column + " (COL)"

    result = summary_occurrence(x, column)
    result.pop("Num Documents")
    maxval = result["Cited by"].max()
    if minmax is not None:
        min_value, max_value = minmax
        if min_value is not None:
            if min_value > maxval:
                min_value = maxval
            result = result[result["Cited by"] >= min_value]
        if max_value is not None:
            if max_value > maxval:
                max_value = maxval
            result = result[result["Cited by"] <= max_value]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[column_IDX].map(lambda w: w in keywords)]
        result = result[result[column_COL].map(lambda w: w in keywords)]
    result.sort_values(
        ["Cited by", column_IDX, column_COL],
        ascending=[False, True, True],
        inplace=True,
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result,
            values="Cited by",
            index=column_IDX,
            columns=column_COL,
            fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    if retmaxval is True:
        return result, maxval
    return result


#
#
#  Co-occurrence
#
#


def summary_co_occurrence(x, column_IDX, column_COL, keywords=None):
    """Summary occurrence and citations by terms in two different columns.

    Args:
        column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
        column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column_COL.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author Keywords': y,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author Keywords  Cited by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4

    >>> summary_co_occurrence(df, column_IDX='Authors', column_COL='Author Keywords')
      Authors (IDX) Author Keywords (COL)  Num Documents  Cited by      ID
    0             A                     a              2         1  [0, 1]
    1             A                     b              1         1     [1]
    2             A                     c              1         3     [3]
    3             B                     a              1         1     [1]
    4             B                     b              2         3  [1, 2]
    5             B                     c              2         7  [3, 4]
    6             B                     d              1         4     [4]
    7             C                     c              1         3     [3]
    8             D                     c              1         4     [4]
    9             D                     d              1         4     [4]

    >>> keywords = Keywords(['B', 'C', 'a', 'b'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> summary_co_occurrence(df, 'Authors', 'Author Keywords', keywords=keywords)
      Authors (IDX) Author Keywords (COL)  Num Documents  Cited by      ID
    0             B                     a              1         1     [1]
    1             B                     b              2         3  [1, 2]


    """

    def generate_pairs(w, v):
        if column_IDX in MULTIVALUED_COLS:
            w = [x.strip() for x in w.split(";")]
        else:
            w = [w]
        if column_COL in MULTIVALUED_COLS:
            v = [x.strip() for x in v.split(";")]
        else:
            v = [v]
        result = []
        for idx0 in range(len(w)):
            for idx1 in range(len(v)):
                result.append((w[idx0], v[idx1]))
        return result

    if column_IDX == column_COL:
        return summary_occurrence(x, column_IDX, keywords)
    data = x.copy()
    data = data[[column_IDX, column_COL, "Cited by", "ID"]]
    data = data.dropna()
    data["Num Documents"] = 1
    data["pairs"] = [
        generate_pairs(a, b) for a, b in zip(data[column_IDX], data[column_COL])
    ]
    data = data[["pairs", "Num Documents", "Cited by", "ID"]]
    data = data.explode("pairs")
    result = data.groupby("pairs", as_index=False).agg(
        {"Cited by": np.sum, "Num Documents": np.sum, "ID": list}
    )
    result["Cited by"] = result["Cited by"].map(int)
    result[column_IDX + " (IDX)"] = result["pairs"].map(lambda x: x[0])
    result[column_COL + " (COL)"] = result["pairs"].map(lambda x: x[1])
    result.pop("pairs")
    result = result[
        [
            column_IDX + " (IDX)",
            column_COL + " (COL)",
            "Num Documents",
            "Cited by",
            "ID",
        ]
    ]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[column_IDX + " (IDX)"].map(lambda w: w in keywords)]
        result = result[result[column_COL + " (COL)"].map(lambda w: w in keywords)]
    result = result.sort_values(
        [column_IDX + " (IDX)", column_COL + " (COL)"], ignore_index=True,
    )
    return result


def co_occurrence(
    x,
    column_IDX,
    column_COL,
    as_matrix=False,
    minmax=None,
    keywords=None,
    retmaxval=False,
):
    """Computes the co-occurrence of two terms in different colums. The report adds
    the number of documents by term between brackets.

    Args:
        column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
        column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column_COL.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author Keywords': y,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author Keywords  Cited by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4

    >>> co_occurrence(df, column_IDX='Authors', column_COL='Author Keywords')
      Authors (IDX) Author Keywords (COL)  Num Documents      ID
    0             A                     a              2  [0, 1]
    1             B                     b              2  [1, 2]
    2             B                     c              2  [3, 4]
    3             A                     b              1     [1]
    4             A                     c              1     [3]
    5             B                     a              1     [1]
    6             B                     d              1     [4]
    7             C                     c              1     [3]
    8             D                     c              1     [4]
    9             D                     d              1     [4]

    >>> co_occurrence(df, column_IDX='Authors', column_COL='Author Keywords', as_matrix=True)
       a  b  c  d
    A  2  1  1  0
    B  1  2  2  1
    C  0  0  1  0
    D  0  0  1  1

    >>> co_occurrence(df, column_IDX='Authors', column_COL='Author Keywords', as_matrix=True, minmax=(2,2))
       a  b  c
    A  2  0  0
    B  0  2  2

    >>> keywords = Keywords(['A', 'B', 'c', 'd'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> co_occurrence(df, 'Authors', 'Author Keywords', as_matrix=True, keywords=keywords)
       c  d
    A  1  0
    B  2  1


    """

    def generate_dic(column, sep):
        new_names = documents_by_term(x, column)
        new_names = {
            term: "{:s} [{:d}]".format(term, docs_per_term)
            for term, docs_per_term in zip(
                new_names[column], new_names["Num Documents"],
            )
        }
        return new_names

    if column_IDX == column_COL:
        return occurrence(
            x,
            column_IDX,
            as_matrix=as_matrix,
            minmax=minmax,
            keywords=keywords,
            retmaxval=retmaxval,
        )
    result = summary_co_occurrence(x, column_IDX, column_COL, keywords)
    result.pop("Cited by")
    maxval = result["Num Documents"].max()
    if minmax is not None:
        min_value, max_value = minmax
        if min_value is not None:
            if min_value > maxval:
                min_value = maxval
            result = result[result["Num Documents"] >= min_value]
        if max_value is not None:
            if max_value > maxval:
                max_value = maxval
            result = result[result["Num Documents"] <= max_value]
    result.sort_values(
        [column_IDX + " (IDX)", column_COL + " (COL)", "Num Documents",],
        ascending=[True, True, False],
        inplace=True,
    )
    result = result.sort_values(
        ["Num Documents", column_IDX + " (IDX)", column_COL + " (COL)"],
        ascending=[False, True, True],
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result,
            values="Num Documents",
            index=column_IDX + " (IDX)",
            columns=column_COL + " (COL)",
            fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    if retmaxval is True:
        return result, maxval
    return result


def co_citation(
    x,
    column_IDX,
    column_COL,
    as_matrix=False,
    minmax=None,
    keywords=None,
    retmaxval=False,
):
    """Computes the number of citations shared by two terms in different columns.

    Args:
        column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
        column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column_COL.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (Keywords): filter the result using the specified Keywords object.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author Keywords': y,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author Keywords  Cited by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4

    >>> co_citation(df, column_IDX='Authors', column_COL='Author Keywords')
      Authors (IDX) Author Keywords (COL)  Cited by      ID
    0             B                     c         7  [3, 4]
    1             B                     d         4     [4]
    2             D                     c         4     [4]
    3             D                     d         4     [4]
    4             A                     c         3     [3]
    5             B                     b         3  [1, 2]
    6             C                     c         3     [3]
    7             A                     a         1  [0, 1]
    8             A                     b         1     [1]
    9             B                     a         1     [1]

    >>> co_citation(df, column_IDX='Authors', column_COL='Author Keywords', as_matrix=True)
       a  b  c  d
    A  1  1  3  0
    B  1  3  7  4
    C  0  0  3  0
    D  0  0  4  4

    >>> co_citation(df, column_IDX='Authors', column_COL='Author Keywords', as_matrix=True, minmax=(3,4))
       b  c  d
    A  0  3  0
    B  3  0  4
    C  0  3  0
    D  0  4  4

    >>> keywords = Keywords(['A', 'B', 'c', 'd'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> co_citation(df, 'Authors', 'Author Keywords', as_matrix=True, keywords=keywords)
       c  d
    A  3  0
    B  7  4

    """

    def generate_dic(column):
        new_names = citations_by_term(x, column)
        new_names = {
            term: "{:s} [{:d}]".format(term, docs_per_term)
            for term, docs_per_term in zip(new_names[column], new_names["Cited by"],)
        }
        return new_names

    result = summary_co_occurrence(x, column_IDX, column_COL, keywords)
    result.pop("Num Documents")
    maxval = result["Cited by"].max()
    if minmax is not None:
        min_value, max_value = minmax
        if min_value is not None:
            result = result[result["Cited by"] >= min_value]
        if max_value is not None:
            result = result[result["Cited by"] <= max_value]
    result.sort_values(
        ["Cited by", column_IDX + " (IDX)", column_COL + " (COL)",],
        ascending=[False, True, True,],
        inplace=True,
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result,
            values="Cited by",
            index=column_IDX + " (IDX)",
            columns=column_COL + " (COL)",
            fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    if retmaxval is True:
        return result, retmaxval
    return result


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
