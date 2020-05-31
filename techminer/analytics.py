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


# def summary_by_term(x, column, keywords=None):
#     """Summarize the number of documents and citations by term in a dataframe.

#     Args:
#         column (str): the column to explode.
#         keywords (int, list): filter the results.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...     {
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
#     ...          "Cited by": list(range(10,14)),
#     ...          "ID": list(range(4)),
#     ...     }
#     ... )
#     >>> x
#                           Authors  Cited by  ID
#     0  author 0;author 1;author 2        10   0
#     1                    author 0        11   1
#     2                    author 1        12   2
#     3                    author 3        13   3

#     >>> summary_by_term(x, 'Authors')
#         Authors  Num Documents  Cited by      ID
#     0  author 0              2        21  [0, 1]
#     1  author 1              2        22  [0, 2]
#     2  author 2              1        10     [0]
#     3  author 3              1        13     [3]

#     >>> keywords = Keywords(['author 1', 'author 2'])
#     >>> keywords = keywords.compile()
#     >>> summary_by_term(x, 'Authors', keywords=keywords)
#         Authors  Num Documents  Cited by      ID
#     0  author 1              2        22  [0, 2]
#     1  author 2              1        10     [0]

#     """
#     x = x.copy()
#     x = __explode(x[[column, "Cited by", "ID"]], column)
#     x["Num Documents"] = 1
#     result = x.groupby(column, as_index=False).agg(
#         {"Num Documents": np.size, "Cited by": np.sum}
#     )
#     result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
#     result["Cited by"] = result["Cited by"].map(lambda x: int(x))
#     if keywords is not None:
#         result = result[result[column].map(lambda w: w in keywords)]
#     result.sort_values(
#         [column, "Num Documents", "Cited by"],
#         ascending=[True, False, False],
#         inplace=True,
#         ignore_index=True,
#     )
#     return result


# def documents_by_term(x, column, keywords=None):
#     """Computes the number of documents per term in a given column.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...     {
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
#     ...          "Cited by": list(range(10,14)),
#     ...          "ID": list(range(4)),
#     ...     }
#     ... )
#     >>> x
#                           Authors  Cited by  ID
#     0  author 0;author 1;author 2        10   0
#     1                    author 0        11   1
#     2                    author 1        12   2
#     3                    author 3        13   3

#     >>> documents_by_term(x, 'Authors')
#         Authors  Num Documents      ID
#     0  author 0              2  [0, 1]
#     1  author 1              2  [0, 2]
#     2  author 2              1     [0]
#     3  author 3              1     [3]

#     >>> keywords = Keywords(['author 1', 'author 2'])
#     >>> keywords = keywords.compile()
#     >>> documents_by_term(x, 'Authors', keywords=keywords)
#         Authors  Num Documents      ID
#     0  author 1              2  [0, 2]
#     1  author 2              1     [0]

#     """

#     result = summary_by_term(x, column, keywords)
#     result.pop("Cited by")
#     result.sort_values(
#         ["Num Documents", column],
#         ascending=[False, True],
#         inplace=True,
#         ignore_index=True,
#     )
#     return result


# def citations_by_term(x, column, keywords=None):
#     """Computes the number of citations by item in a column.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...     {
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
#     ...          "Cited by": list(range(10,14)),
#     ...          "ID": list(range(4)),
#     ...     }
#     ... )
#     >>> x
#                           Authors  Cited by  ID
#     0  author 0;author 1;author 2        10   0
#     1                    author 0        11   1
#     2                    author 1        12   2
#     3                    author 3        13   3

#     >>> citations_by_term(x, 'Authors')
#         Authors  Cited by      ID
#     0  author 1        22  [0, 2]
#     1  author 0        21  [0, 1]
#     2  author 3        13     [3]
#     3  author 2        10     [0]

#     >>> keywords = Keywords(['author 1', 'author 2'])
#     >>> keywords = keywords.compile()
#     >>> citations_by_term(x, 'Authors', keywords=keywords)
#         Authors  Cited by      ID
#     0  author 1        22  [0, 2]
#     1  author 2        10     [0]


#     """
#     result = summary_by_term(x, column, keywords)
#     result.pop("Num Documents")
#     result.sort_values(
#         ["Cited by", column], ascending=[False, True], inplace=True, ignore_index=True,
#     )
#     return result


# def summary_by_term_per_year(x, column, keywords=None):
#     """Computes the number of documents and citations by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> summary_by_term_per_year(df, 'Authors')
#         Authors  Year  Cited by  Num Documents      ID
#     0  author 0  2010        21              2  [0, 1]
#     1  author 1  2010        10              1     [0]
#     2  author 2  2010        10              1     [0]
#     3  author 1  2011        12              1     [2]
#     4  author 3  2011        13              1     [3]
#     5  author 4  2012        14              1     [4]
#     6  author 4  2014        15              1     [5]

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> summary_by_term_per_year(df, 'Authors', keywords=keywords)
#         Authors  Year  Cited by  Num Documents   ID
#     0  author 1  2010        10              1  [0]
#     1  author 2  2010        10              1  [0]
#     2  author 1  2011        12              1  [2]
#     3  author 3  2011        13              1  [3]

#     """
#     data = __explode(x[["Year", column, "Cited by", "ID"]], column)
#     data["Num Documents"] = 1
#     result = data.groupby([column, "Year"], as_index=False).agg(
#         {"Cited by": np.sum, "Num Documents": np.size}
#     )
#     result = result.assign(
#         ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
#     )
#     result["Cited by"] = result["Cited by"].map(lambda x: int(x))
#     if keywords is not None:
#         if keywords._patterns is None:
#             keywords = keywords.compile()
#         result = result[result[column].map(lambda w: w in keywords)]
#     result.sort_values(
#         ["Year", column], ascending=True, inplace=True, ignore_index=True,
#     )
#     return result


# def documents_by_term_per_year(x, column, as_matrix=False, minmax=None, keywords=None):
#     """Computes the number of documents by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         as_matrix (bool): Results are returned as a matrix.
#         minmax (pair(number,number)): filter values by >=min,<=max.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> documents_by_term_per_year(df, 'Authors')
#         Authors  Year  Num Documents      ID
#     0  author 0  2010              2  [0, 1]
#     1  author 1  2010              1     [0]
#     2  author 2  2010              1     [0]
#     3  author 1  2011              1     [2]
#     4  author 3  2011              1     [3]
#     5  author 4  2012              1     [4]
#     6  author 4  2014              1     [5]

#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         2         1         1         0         0
#     2011         0         1         0         1         0
#     2012         0         0         0         0         1
#     2014         0         0         0         0         1

#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True, minmax=(2, None))
#           author 0
#     2010         2

#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True, minmax=(0, 1))
#           author 1  author 2  author 3  author 4
#     2010         1         1         0         0
#     2011         1         0         1         0
#     2012         0         0         0         1
#     2014         0         0         0         1

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> documents_by_term_per_year(df, 'Authors', keywords=keywords, as_matrix=True)
#           author 1  author 2  author 3
#     2010         1         1         0
#     2011         1         0         1

#     """

#     result = summary_by_term_per_year(x, column, keywords)
#     result.pop("Cited by")
#     if minmax is not None:
#         min_value, max_value = minmax
#         if min_value is not None:
#             result = result[result["Num Documents"] >= min_value]
#         if max_value is not None:
#             result = result[result["Num Documents"] <= max_value]
#     result.sort_values(
#         ["Year", "Num Documents", column], ascending=[True, False, True], inplace=True,
#     )
#     result.reset_index(drop=True)
#     if as_matrix == True:
#         result = pd.pivot_table(
#             result, values="Num Documents", index="Year", columns=column, fill_value=0,
#         )
#         result.columns = result.columns.tolist()
#         result.index = result.index.tolist()
#     return result


# def gant(x, column, minmax=None, keywords=None):
#     """Computes the number of documents by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         as_matrix (bool): Results are returned as a matrix.
#         minmax (pair(number,number)): filter values by >=min,<=max.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2011, 2011, 2012, 2015, 2012, 2016],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 3,author 4,author 4".split(","),
#     ...          "Cited by": list(range(10,17)),
#     ...          "ID": list(range(7)),
#     ...     }
#     ... )
#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         1         1         0         0
#     2011         1         1         0         0         0
#     2012         0         0         0         1         1
#     2015         0         0         0         1         0
#     2016         0         0         0         0         1

#     >>> gant(df, 'Authors')
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         1         1         0         0
#     2011         1         1         0         0         0
#     2012         0         0         0         1         1
#     2013         0         0         0         1         1
#     2014         0         0         0         1         1
#     2015         0         0         0         1         1
#     2016         0         0         0         0         1

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> gant(df, 'Authors', keywords=keywords)
#           author 1  author 2  author 3
#     2010         1         1         0
#     2011         1         0         0
#     2012         0         0         1
#     2013         0         0         1
#     2014         0         0         1
#     2015         0         0         1

#     """
#     result = documents_by_term_per_year(
#         x, column=column, as_matrix=True, minmax=minmax, keywords=keywords
#     )
#     years = [year for year in range(result.index.min(), result.index.max() + 1)]
#     result = result.reindex(years, fill_value=0)
#     matrix1 = result.copy()
#     matrix1 = matrix1.cumsum()
#     matrix1 = matrix1.applymap(lambda x: True if x > 0 else False)
#     matrix2 = result.copy()
#     matrix2 = matrix2.sort_index(ascending=False)
#     matrix2 = matrix2.cumsum()
#     matrix2 = matrix2.applymap(lambda x: True if x > 0 else False)
#     matrix2 = matrix2.sort_index(ascending=True)
#     result = matrix1.eq(matrix2)
#     result = result.applymap(lambda x: 1 if x is True else 0)
#     return result


# def citations_by_term_per_year(x, column, as_matrix=False, minmax=None, keywords=None):
#     """Computes the number of citations by term by year in a column.

#     Args:
#         column (str): the column to explode.
#         as_matrix (bool): Results are returned as a matrix.
#         minmax (pair(number,number)): filter values by >=min,<=max.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> citations_by_term_per_year(df, 'Authors')
#         Authors  Year  Cited by      ID
#     0  author 0  2010        21  [0, 1]
#     1  author 2  2010        10     [0]
#     2  author 1  2010        10     [0]
#     3  author 3  2011        13     [3]
#     4  author 1  2011        12     [2]
#     5  author 4  2012        14     [4]
#     6  author 4  2014        15     [5]

#     >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010        21        10        10         0         0
#     2011         0        12         0        13         0
#     2012         0         0         0         0        14
#     2014         0         0         0         0        15

#     >>> citations_by_term_per_year(df, 'Authors', as_matrix=True, minmax=(12, 15))
#           author 1  author 3  author 4
#     2011        12        13         0
#     2012         0         0        14
#     2014         0         0        15

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> citations_by_term_per_year(df, 'Authors', keywords=keywords)
#         Authors  Year  Cited by   ID
#     0  author 2  2010        10  [0]
#     1  author 1  2010        10  [0]
#     2  author 3  2011        13  [3]
#     3  author 1  2011        12  [2]

#     """
#     result = summary_by_term_per_year(x, column, keywords)
#     result.pop("Num Documents")
#     if minmax is not None:
#         min_value, max_value = minmax
#         if min_value is not None:
#             result = result[result["Cited by"] >= min_value]
#         if max_value is not None:
#             result = result[result["Cited by"] <= max_value]
#     result.sort_values(
#         ["Year", "Cited by", column], ascending=[True, False, False], inplace=True,
#     )
#     result = result.reset_index(drop=True)
#     if as_matrix == True:
#         result = pd.pivot_table(
#             result, values="Cited by", index="Year", columns=column, fill_value=0,
#         )
#         result.columns = result.columns.tolist()
#         result.index = result.index.tolist()
#     return result


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
