
"""
Primary Data Importation and Manipulation
==================================================================================================



"""

import logging

import numpy as np
import pandas as pd
from techminer.explode import MULTIVALUED_COLS
from techminer.text import extract_country, extract_institution, remove_accents

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
