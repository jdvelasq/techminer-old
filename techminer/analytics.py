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
    #
    "Affiliations_",
    "Author Keywords_",
    "Author(s) Country_",
    "Author(s) ID_",
    "Author(s) Institution_",
    "Authors with affiliations_",
    "Authors_",
    "Countries_",
    "Index Keywords_",
    "Institutions_",
    "Keywords_",
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


def __disambiguate_authors(x):
    """Verify if author's names are unique. For duplicated names, based on `Author(s) ID` column,
    adds a consecutive number to the name.


    Args:
        col_Authors (str): Author's name column.
        sep_Authors (str): Character used as internal separator for the elements in the column with the author's name.
        col_AuthorsID (str): Author's ID column.
        sep_AuthorsID (str): Character used as internal separator for the elements in the column with the author's ID.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...   {
    ...      "Authors": "author 0;author 0;author 0,author 0,author 0".split(","),
    ...      "Author(s) ID": "0;1;2;,3;,4;".split(','),
    ...   }
    ... )
    >>> df
                          Authors Author(s) ID
    0  author 0;author 0;author 0       0;1;2;
    1                    author 0           3;
    2                    author 0           4;

    >>> __disambiguate_authors(df)
                                Authors Author(s) ID
    0  author 0;author 0(1);author 0(2)        0;1;2
    1                       author 0(3)            3
    2                       author 0(4)            4


    """

    x["Authors"] = x["Authors"].map(
        lambda w: w[:-1] if not pd.isna(w) and w[-1] == ";" else w
    )

    x["Author(s) ID"] = x["Author(s) ID"].map(
        lambda w: w[:-1] if not pd.isna(w) and w[-1] == ";" else w
    )

    data = x[["Authors", "Author(s) ID"]]
    data = data.dropna()

    data["*info*"] = [(a, b) for (a, b) in zip(data["Authors"], data["Author(s) ID"])]

    data["*info*"] = data["*info*"].map(
        lambda x: [
            (u.strip(), v.strip()) for u, v in zip(x[0].split(";"), x[1].split(";"))
        ]
    )

    data = data[["*info*"]].explode("*info*")
    data = data.reset_index(drop=True)

    names_ids = {}
    for idx in range(len(data)):

        author_name = data.at[idx, "*info*"][0]
        author_id = data.at[idx, "*info*"][1]

        if author_name in names_ids.keys():

            if author_id not in names_ids[author_name]:
                names_ids[author_name] = names_ids[author_name] + [author_id]
        else:
            names_ids[author_name] = [author_id]

    ids_names = {}
    for author_name in names_ids.keys():
        suffix = 0
        for author_id in names_ids[author_name]:
            if suffix > 0:
                ids_names[author_id] = author_name + "(" + str(suffix) + ")"
            else:
                ids_names[author_id] = author_name
            suffix += 1

    result = x.copy()

    result["Authors"] = result["Author(s) ID"].map(
        lambda z: ";".join([ids_names[w.strip()] for w in z.split(";")])
        if not pd.isna(z)
        else z
    )

    return result


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
    x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)
    if "Authors" in x.columns:
        #
        logging.info("Formatting author names ...")
        #
        x["Authors"] = x.Authors.map(
            lambda w: w.replace(",", ";").replace(".", "") if pd.isna(w) is False else w
        )
        x["Authors"] = x.Authors.map(
            lambda w: pd.NA if w == "[No author name available]" else w
        )
        x["Author(s) ID"] = x["Author(s) ID"].map(
            lambda w: pd.NA if w == "[No author id available]" else w
        )
    #
    if "Authors" in x.columns:
        #
        logging.info("Disambiguating author names ...")
        #
        x = __disambiguate_authors(x)
    #
    if "Title" in x.columns:
        #
        logging.info("Removing part of titles in foreing languages ...")
        #
        x["Title"] = x.Title.map(
            lambda w: w[0 : w.find("[")] if pd.isna(w) is False and w[-1] == "]" else w
        )
    #
    if (
        "Author Keywords" in x.columns.tolist()
        and "Index Keywords" in x.columns.tolist()
    ):
        #
        logging.info("Fusioning author and index keywords ...")
        #
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
    if "Affiliations" in x.columns:
        #
        logging.info("Extracting countries from affiliations ...")
        #
        x["Countries"] = x.Affiliations.map(
            lambda w: extract_country(w) if pd.isna(w) is False else w
        )
    #
    if "Affiliations" in x.columns:
        #
        logging.info("Extracting institutions from affiliations ...")
        #
        x["Institutions"] = x.Affiliations.map(
            lambda w: extract_institution(w) if pd.isna(w) is False else w
        )
    #
    if "Countries" in x.columns:
        #
        logging.info("Extracting country of 1st author ...")
        #
        x["Country 1st"] = x["Countries"].map(
            lambda w: w.split(";")[0] if not pd.isna(w) else w
        )
    #
    if "Institutions" in x.columns:
        #
        logging.info("Extracting affiliation of 1st author ...")
        #
        x["Institution 1st"] = x["Institutions"].map(
            lambda w: w.split(";")[0] if not pd.isna(w) else w
        )
    #
    if "Authors" in x.columns:
        #
        logging.info("Counting number of authors ...")
        x["Num Authors"] = x["Authors"].map(
            lambda w: len(w.split(";")) if not pd.isna(w) else 0
        )
    #
    x["ID"] = range(len(x))
    #
    x = x.applymap(lambda w: pd.NA if isinstance(w, str) and w == "" else w)
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
    cagr = str(round(100 * (np.power(Pn / Po, 1 / n) - 1), 2)) + " %"
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


def documents_by_term_per_year(x, column, as_matrix=False, keywords=None):
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

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> documents_by_term_per_year(df, 'Authors', keywords=keywords, as_matrix=True)
          author 1  author 2  author 3
    2010         1         1         0
    2011         1         0         1

    """

    result = summary_by_term_per_year(x, column, keywords)
    result.pop("Cited by")
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


def gant(x, column, keywords=None):
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
        x, column=column, as_matrix=True, keywords=keywords
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


def citations_by_term_per_year(x, column, as_matrix=False, keywords=None):
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


#
#
# Occurrence and co-occurrence
#
#


def summary_co_occurrence(x, column, by=None, keywords=None):
    """Summary occurrence and citations by terms in two different columns.

    Args:
        by (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the by.
        column (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column.
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

    >>> summary_co_occurrence(df, column='Author Keywords', by='Authors')
      Author Keywords Authors  Num Documents  Cited by      ID
    0               a       A              2         1  [0, 1]
    1               a       B              1         1     [1]
    2               b       A              1         1     [1]
    3               b       B              2         3  [1, 2]
    4               c       A              1         3     [3]
    5               c       B              2         7  [3, 4]
    6               c       C              1         3     [3]
    7               c       D              1         4     [4]
    8               d       B              1         4     [4]
    9               d       D              1         4     [4]

    >>> keywords = Keywords(['B', 'C', 'a', 'b'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> summary_co_occurrence(df, 'Authors', 'Author Keywords', keywords=keywords)
      Authors Author Keywords  Num Documents  Cited by      ID
    0       B               a              1         1     [1]
    1       B               b              2         3  [1, 2]


    """

    def generate_pairs(w, v):
        if by in MULTIVALUED_COLS:
            w = [x.strip() for x in w.split(";")]
        else:
            w = [w]
        if column in MULTIVALUED_COLS:
            v = [x.strip() for x in v.split(";")]
        else:
            v = [v]
        result = []
        for idx0 in range(len(w)):
            for idx1 in range(len(v)):
                result.append((w[idx0], v[idx1]))
        return result

    #
    data = x.copy()
    if by is None or by == column:
        by = column + "_"
        data[by] = data[column].copy()

    data = data[[by, column, "Cited by", "ID"]]
    data = data.dropna()
    data["Num Documents"] = 1
    data["pairs"] = [generate_pairs(a, b) for a, b in zip(data[by], data[column])]
    data = data[["pairs", "Num Documents", "Cited by", "ID"]]
    data = data.explode("pairs")
    result = data.groupby("pairs", as_index=False).agg(
        {"Cited by": np.sum, "Num Documents": np.sum, "ID": list}
    )
    result["Cited by"] = result["Cited by"].map(int)
    result[by] = result["pairs"].map(lambda x: x[0])
    result[column] = result["pairs"].map(lambda x: x[1])
    result.pop("pairs")
    result = result[[column, by, "Num Documents", "Cited by", "ID",]]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[by].map(lambda w: w in keywords)]
        result = result[result[column].map(lambda w: w in keywords)]
    result = result.sort_values([column, by], ignore_index=True,)
    return result


def co_occurrence(
    x, column, by=None, as_matrix=False, min_value=0, keywords=None,
):
    """Computes the co-occurrence of two terms in different colums. The report adds
    the number of documents by term between brackets.

    Args:
        by (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the by.
        column (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column.
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

    >>> co_occurrence(df, column='Authors', by='Author Keywords')
      Authors Author Keywords  Num Documents      ID
    0       A               a              2  [0, 1]
    1       B               b              2  [1, 2]
    2       B               c              2  [3, 4]
    3       A               b              1     [1]
    4       A               c              1     [3]
    5       B               a              1     [1]
    6       B               d              1     [4]
    7       C               c              1     [3]
    8       D               c              1     [4]
    9       D               d              1     [4]

    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True)
       a  b  c  d
    A  2  1  1  0
    B  1  2  2  1
    C  0  0  1  0
    D  0  0  1  1

    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True, min_value=2)
       a  b  c
    A  2  1  1
    B  1  2  2

    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True, min_value=5)
       a  b  c
    A  2  1  1
    B  1  2  2

    >>> keywords = Keywords(['A', 'B', 'c', 'd'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True, keywords=keywords)
       c  d
    A  1  0
    B  2  1

    

    """

    # def generate_dic(column, sep):
    #     new_names = documents_by_term(x, column)
    #     new_names = {
    #         term: "{:s} [{:d}]".format(term, docs_per_term)
    #         for term, docs_per_term in zip(
    #             new_names[column], new_names["Num Documents"],
    #         )
    #     }
    #     return new_names

    #
    result = summary_co_occurrence(x, column=column, by=by, keywords=keywords)
    if by is None or by == column:
        by = column + "_"
    result.pop("Cited by")
    #
    if as_matrix is False:
        result = result.sort_values(
            ["Num Documents", column, by], ascending=[False, True, True],
        )
        if min_value is not None and min_value > 0:
            result = result[result["Num Documents"] >= min_value]
        result = result.reset_index(drop=True)
        return result
    #
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Num Documents", index=by, columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    if min_value is not None and min_value > 0:
        #
        a = result.max(axis=1)
        b = result.max(axis=0)
        a = a.sort_values(ascending=False)
        b = b.sort_values(ascending=False)
        min_value = (
            min(a.max(), b.max()) if min_value > min(a.max(), b.max()) else min_value
        )
        a = a[a >= min_value]
        b = b[b >= min_value]
        #
        result = result.loc[sorted(a.index), sorted(b.index)]
    return result


#
#
#


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
       Authors Authors_  Num Documents  Cited by            ID
    0        A        A              4         7  [0, 1, 2, 4]
    1        A        B              2         6        [2, 4]
    2        A        C              1         4           [4]
    3        B        A              2         6        [2, 4]
    4        B        B              4        15  [2, 3, 4, 6]
    5        B        C              1         4           [4]
    6        B        D              1         6           [6]
    7        C        A              1         4           [4]
    8        C        B              1         4           [4]
    9        C        C              1         4           [4]
    10       D        B              1         6           [6]
    11       D        D              2        11        [5, 6]

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> summary_occurrence(df, 'Authors', keywords=keywords)
      Authors Authors_  Num Documents  Cited by            ID
    0       A        A              4         7  [0, 1, 2, 4]
    1       A        B              2         6        [2, 4]
    2       B        A              2         6        [2, 4]
    3       B        B              4        15  [2, 3, 4, 6]


    """
    return summary_co_occurrence(x=x, column=column, by=None, keywords=keywords)


def occurrence(x, column, as_matrix=False, min_value=0, keywords=None):
    """Computes the occurrence between the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
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
       Authors Authors_  Num Documents            ID
    0        A        A              4  [0, 1, 2, 4]
    1        B        B              4  [2, 3, 4, 6]
    2        A        B              2        [2, 4]
    3        B        A              2        [2, 4]
    4        D        D              2        [5, 6]
    5        A        C              1           [4]
    6        B        C              1           [4]
    7        B        D              1           [6]
    8        C        A              1           [4]
    9        C        B              1           [4]
    10       C        C              1           [4]
    11       D        B              1           [6]

    >>> occurrence(df, column='Authors', as_matrix=True)
       A  B  C  D
    A  4  2  1  0
    B  2  4  1  1
    C  1  1  1  0
    D  0  1  0  2

    >>> occurrence(df, column='Authors', min_value=2, as_matrix=True)
       A  B  D
    A  4  2  0
    B  2  4  1
    D  0  1  2

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> occurrence(df, 'Authors', as_matrix=True, keywords=keywords)
       A  B
    A  4  2
    B  2  4

    """
    return co_occurrence(
        x,
        column=column,
        by=None,
        as_matrix=as_matrix,
        min_value=min_value,
        keywords=keywords,
    )


#
#
#  Analytical functions
#
#


def compute_tfm(x, column, keywords=None):
    """Computes the term-frequency matrix for the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
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

    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1

    >>> compute_tfm(df, 'Author Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> compute_tfm(df, 'Authors', keywords=keywords)
       A  B
    0  1  0
    1  1  1
    2  0  1
    3  1  1
    4  0  1

    """
    data = x[[column, "ID"]].copy()
    data["value"] = 1.0
    data = __explode(data, column)
    if keywords is not None:
        if isinstance(keywords, list):
            keywords = Keywords(keywords, ignore_case=False, full_match=True)
        if keywords._patterns is None:
            keywords = keywords.compile()
        data = data[data[column].map(lambda w: w in keywords)]
    result = pd.pivot_table(
        data=data, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)
    return result


def corr(
    x,
    column,
    by=None,
    method="pearson",
    min_link_value=-1,
    cmap=None,
    as_matrix=True,
    filter_by=None,
):
    """Computes cross-correlation among items in two different columns of the dataframe.

    Args:
        column_IDX (str): the first column.
        sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
        column_COL (str): the second column.
        sep_COL (str): Character used as internal separator for the elements in the column_COL.
        method (str): Available methods are:

            - pearson : Standard correlation coefficient.

            - kendall : Kendall Tau correlation coefficient.

            - spearman : Spearman rank correlation.

        as_matrix (bool): the result is reshaped by melt or not.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
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
    5     A;B               d         5   5


    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> compute_tfm(df, 'Author Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1
    5  0  0  0  1


    >>> corr(df, 'Authors', 'Author Keywords')
              A         B         C        D
    A  1.000000 -1.000000 -0.333333 -0.57735
    B -1.000000  1.000000  0.333333  0.57735
    C -0.333333  0.333333  1.000000  0.57735
    D -0.577350  0.577350  0.577350  1.00000

    >>> corr(df, 'Authors', 'Author Keywords', min_link_value=0)
              B         C        D
    B  1.000000  0.333333  0.57735
    C  0.333333  1.000000  0.57735
    D  0.577350  0.577350  1.00000

    >>> corr(df, 'Authors', 'Author Keywords', as_matrix=False)
       Authors Author Keywords     value
    0        A               A  1.000000
    1        B               A -1.000000
    2        C               A -0.333333
    3        D               A -0.577350
    4        A               B -1.000000
    5        B               B  1.000000
    6        C               B  0.333333
    7        D               B  0.577350
    8        A               C -0.333333
    9        B               C  0.333333
    10       C               C  1.000000
    11       D               C  0.577350
    12       A               D -0.577350
    13       B               D  0.577350
    14       C               D  0.577350
    15       D               D  1.000000

    >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> corr(df, 'Authors', 'Author Keywords', filter_by=keywords)
              A         B         C
    A  1.000000 -1.000000 -0.333333
    B -1.000000  1.000000  0.333333
    C -0.333333  0.333333  1.000000

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
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
    5     A;B               d         5   5

    >>> compute_tfm(df, column='Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> corr(df, 'Authors')
              A         B         C         D
    A  1.000000 -0.316228  0.316228 -0.632456
    B -0.316228  1.000000  0.200000  0.200000
    C  0.316228  0.200000  1.000000 -0.200000
    D -0.632456  0.200000 -0.200000  1.000000

    >>> corr(df, 'Authors', as_matrix=False)
       Authors Authors_     value
    0        A        A  1.000000
    1        B        A -0.316228
    2        C        A  0.316228
    3        D        A -0.632456
    4        A        B -0.316228
    5        B        B  1.000000
    6        C        B  0.200000
    7        D        B  0.200000
    8        A        C  0.316228
    9        B        C  0.200000
    10       C        C  1.000000
    11       D        C -0.200000
    12       A        D -0.632456
    13       B        D  0.200000
    14       C        D -0.200000
    15       D        D  1.000000

    >>> corr(df, 'Author Keywords')
          a     b     c     d
    a  1.00  0.25 -0.50 -0.50
    b  0.25  1.00 -0.50 -0.50
    c -0.50 -0.50  1.00  0.25
    d -0.50 -0.50  0.25  1.00

    >>> corr(df, 'Author Keywords', min_link_value=0.249)
          a     b     c     d
    a  1.00  0.25 -0.50 -0.50
    b  0.25  1.00 -0.50 -0.50
    c -0.50 -0.50  1.00  0.25
    d -0.50 -0.50  0.25  1.00


    >>> corr(df, 'Author Keywords', min_link_value=1.0)
          c     d
    c  1.00  0.25
    d  0.25  1.00

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> corr(df, 'Authors', filter_by=keywords)
              A         B
    A  1.000000 -0.316228
    B -0.316228  1.000000


    """
    if by is None:
        by = column
    if column == by:
        tfm = compute_tfm(x, column=column, keywords=filter_by)
        result = tfm.corr(method=method)
    else:
        tfm = co_occurrence(x, column=column, by=by, as_matrix=True, keywords=None,)
        result = tfm.corr(method=method)
    #
    if filter_by is not None:
        filter_by = filter_by.compile()
        new_columns = [w for w in result.columns if w in filter_by]
        new_index = [w for w in result.index if w in filter_by]
        result = result.loc[new_index, new_columns]
    #
    if as_matrix is False:
        if column == by:
            result = (
                result.reset_index()
                .melt("index")
                .rename(columns={"index": column, "variable": column + "_"})
            )
        else:
            result = (
                result.reset_index()
                .melt("index")
                .rename(columns={"index": column, "variable": by})
            )

        result = result[result["value"] >= min_link_value]
        return result

    for col in result.columns.tolist():
        result.at[col, col] = -1.0
    a = result.max()
    if min_link_value > a.max():
        min_link_value = a.max()
    a = a[a >= min_link_value]
    result = result.loc[a.index.tolist(), a.index.tolist()]
    for col in result.columns.tolist():
        result.at[col, col] = 1
    result = result.sort_index(axis=0, ascending=True)
    result = result.sort_index(axis=1, ascending=True)
    return result


# def autocorr_map(
#     x, column, method="pearson", minval=None, top_n_links=None, keywords=None,
# ):
#     """Computes the autocorrelation map among items in a column of the dataframe.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         method (str): Available methods are:

#             * pearson : Standard correlation coefficient.

#             * kendall : Kendall Tau correlation coefficient.

#             * spearman : Spearman rank correlation.

#         minval (float): Minimum autocorrelation value to show links.
#         top_n_links (int): Shows top n links.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = [ 'A', 'A;C', 'B', 'A;B;C', 'B;D', 'A;B', 'A;C']
#     >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd', 'c;d']
#     >>> df = pd.DataFrame(
#     ...    {
#     ...       'Authors': x,
#     ...       'Author Keywords': y,
#     ...       'Cited by': list(range(len(x))),
#     ...       'ID': list(range(len(x))),
#     ...    }
#     ... )
#     >>> df
#       Authors Author Keywords  Cited by  ID
#     0       A               a         0   0
#     1     A;C             a;b         1   1
#     2       B               b         2   2
#     3   A;B;C               c         3   3
#     4     B;D             c;d         4   4
#     5     A;B               d         5   5
#     6     A;C             c;d         6   6


#     >>> corr(df, 'Authors')
#               A         B         C         D
#     A  1.000000 -0.547723  0.547723 -0.645497
#     B -0.547723  1.000000 -0.416667  0.353553
#     C  0.547723 -0.416667  1.000000 -0.353553
#     D -0.645497  0.353553 -0.353553  1.000000


#     >>> autocorr_map(df, 'Authors')
#     {'terms': ['A', 'B', 'C', 'D'], 'edges_75': None, 'edges_50': [('A', 'C')], 'edges_25': [('B', 'D')], 'other_edges': None}

#     >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
#     >>> keywords = keywords.compile()
#     >>> corr(df, 'Authors', filter_by=keywords)
#               A         B         C
#     A  1.000000 -0.547723  0.547723
#     B -0.547723  1.000000 -0.416667
#     C  0.547723 -0.416667  1.000000

#     >>> autocorr_map(df, 'Authors', keywords=keywords)
#     {'terms': ['A', 'B', 'C'], 'edges_75': None, 'edges_50': [('A', 'C')], 'edges_25': None, 'other_edges': None}


#     """

#     matrix = corr(
#         x,
#         column=column,
#         method=method,
#         as_matrix=True,
#         show_between=None,
#         filter_by=keywords,
#     )

#     terms = matrix.columns.tolist()

#     n = len(matrix.columns)
#     edges_75 = []
#     edges_50 = []
#     edges_25 = []
#     other_edges = []

#     if top_n_links is not None:
#         values = matrix.to_numpy()
#         top_value = []
#         for icol in range(n):
#             for irow in range(icol + 1, n):
#                 top_value.append(values[irow, icol])
#         top_value = sorted(top_value, reverse=True)
#         top_value = top_value[top_n_links - 1]
#         if minval is not None:
#             minval = max(minval, top_value)
#         else:
#             minval = top_value

#     for icol in range(n):
#         for irow in range(icol + 1, n):
#             if minval is None or matrix[terms[icol]][terms[irow]] >= minval:
#                 if matrix[terms[icol]][terms[irow]] > 0.75:
#                     edges_75.append((terms[icol], terms[irow]))
#                 elif matrix[terms[icol]][terms[irow]] > 0.50:
#                     edges_50.append((terms[icol], terms[irow]))
#                 elif matrix[terms[icol]][terms[irow]] > 0.25:
#                     edges_25.append((terms[icol], terms[irow]))
#                 elif matrix[terms[icol]][terms[irow]] > 0.0:
#                     other_edges.append((terms[icol], terms[irow]))

#     if len(edges_75) == 0:
#         edges_75 = None
#     if len(edges_50) == 0:
#         edges_50 = None
#     if len(edges_25) == 0:
#         edges_25 = None
#     if len(other_edges) == 0:
#         other_edges = None

#     return dict(
#         terms=terms,
#         edges_75=edges_75,
#         edges_50=edges_50,
#         edges_25=edges_25,
#         other_edges=other_edges,
#     )


# def corr_map(
#     x,
#     column,
#     by,
#     sep=None,
#     sep_by=None,
#     method="pearson",
#     minval=None,
#     top_n_links=None,
#     keywords=None,
# ):
#     """Computes the correlation map among items in a column of the dataframe.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         method (str): Available methods are:

#             * pearson : Standard correlation coefficient.

#             * kendall : Kendall Tau correlation coefficient.

#             * spearman : Spearman rank correlation.

#         minval (float): Minimum autocorrelation value to show links.
#         top_n_links (int): Shows top n links.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = [ 'A', 'A;C', 'B', 'A;B;C', 'B;D', 'A;B', 'A;C']
#     >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd', 'c;d']
#     >>> df = pd.DataFrame(
#     ...    {
#     ...       'Authors': x,
#     ...       'Author Keywords': y,
#     ...       'Cited by': list(range(len(x))),
#     ...       'ID': list(range(len(x))),
#     ...    }
#     ... )
#     >>> df
#       Authors Author Keywords  Cited by  ID
#     0       A               a         0   0
#     1     A;C             a;b         1   1
#     2       B               b         2   2
#     3   A;B;C               c         3   3
#     4     B;D             c;d         4   4
#     5     A;B               d         5   5
#     6     A;C             c;d         6   6


#     >>> co_occurrence(df, 'Author Keywords', 'Authors', as_matrix=True)
#        A  B  C  D
#     a  2  0  1  0
#     b  1  1  1  0
#     c  2  2  2  1
#     d  2  2  1  1

#     >>> corr(df, 'Authors', 'Author Keywords')
#               A         B         C         D
#     A  1.000000  0.174078  0.333333  0.577350
#     B  0.174078  1.000000  0.522233  0.904534
#     C  0.333333  0.522233  1.000000  0.577350
#     D  0.577350  0.904534  0.577350  1.000000

#     >>> corr(df, 'Authors', 'Author Keywords')
#               A         B         C         D
#     A  1.000000  0.174078  0.333333  0.577350
#     B  0.174078  1.000000  0.522233  0.904534
#     C  0.333333  0.522233  1.000000  0.577350
#     D  0.577350  0.904534  0.577350  1.000000

#     >>> corr_map(df, 'Authors', 'Author Keywords')
#     {'terms': ['A', 'B', 'C', 'D'], 'edges_75': None, 'edges_50': [('A', 'C')], 'edges_25': [('B', 'D')], 'other_edges': None}

#     >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
#     >>> keywords = keywords.compile()
#     >>> corr_map(df, 'Authors', 'Author Keywords', keywords=keywords)
#     {'terms': ['A', 'B', 'C'], 'edges_75': None, 'edges_50': [('A', 'C')], 'edges_25': None, 'other_edges': None}


#     """

#     matrix = corr(
#         x,
#         column=column,
#         by=column,
#         method=method,
#         as_matrix=True,
#         show_between=None,
#         filter_by=keywords,
#     )

#     terms = matrix.columns.tolist()

#     n = len(matrix.columns)
#     edges_75 = []
#     edges_50 = []
#     edges_25 = []
#     other_edges = []

#     if top_n_links is not None:
#         values = matrix.to_numpy()
#         top_value = []
#         for icol in range(n):
#             for irow in range(icol + 1, n):
#                 top_value.append(values[irow, icol])
#         top_value = sorted(top_value, reverse=True)
#         top_value = top_value[top_n_links - 1]
#         if minval is not None:
#             minval = max(minval, top_value)
#         else:
#             minval = top_value

#     for icol in range(n):
#         for irow in range(icol + 1, n):
#             if minval is None or matrix[terms[icol]][terms[irow]] >= minval:
#                 if matrix[terms[icol]][terms[irow]] > 0.75:
#                     edges_75.append((terms[icol], terms[irow]))
#                 elif matrix[terms[icol]][terms[irow]] > 0.50:
#                     edges_50.append((terms[icol], terms[irow]))
#                 elif matrix[terms[icol]][terms[irow]] > 0.25:
#                     edges_25.append((terms[icol], terms[irow]))
#                 elif matrix[terms[icol]][terms[irow]] > 0.0:
#                     other_edges.append((terms[icol], terms[irow]))

#     if len(edges_75) == 0:
#         edges_75 = None
#     if len(edges_50) == 0:
#         edges_50 = None
#     if len(edges_25) == 0:
#         edges_25 = None
#     if len(other_edges) == 0:
#         other_edges = None

#     return dict(
#         terms=terms,
#         edges_75=edges_75,
#         edges_50=edges_50,
#         edges_25=edges_25,
#         other_edges=other_edges,
#     )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
