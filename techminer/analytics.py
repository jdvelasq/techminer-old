"""
Analytics
==================================================================================================


"""

import pandas as pd
import numpy as np
from techminer.text import remove_accents, extract_country, extract_institution
from tqdm import tqdm
from techminer.keywords import Keywords

SCOPUS_TO_WOS_NAMES = {
    "Abbreviated Source Title": "J9",
    "Abstract": "AB",
    "Access Type": "OA",
    "Affiliations": "C1",
    "Art. No.": "AR",
    "Author Keywords": "DE",
    "Author(s) Country": "AU_CO",
    "Author(s) ID": "RI",
    "Author(s) Institution": "AU_IN",
    "Authors with affiliations": "AU_C1",
    "Authors": "AU",
    "Cited by": "TC",
    "Cited references": "CR",
    "DI": "DOI",
    "Document type": "DT",
    "Editors": "BE",
    "EID": "UT",
    "Index Keywords": "ID",
    "ISBN": "BN",
    "ISSN": "SN",
    "Issue": "IS",
    "Keywords": "AU_ID",
    "Language of Original Document": "LA",
    "Page count": "PG",
    "Page end": "EP",
    "Page start": "BP",
    "Publisher": "PU",
    "PubMed ID": "PM",
    "Source title": "SO",
    "Source": "FN",
    "Subject": "SC",
    "Title": "TI",
    "Volume": "VL",
    "Year": "PY",
}

ABBR_TO_NAMES = {
    "AU": "Authors",
    "DE": "Author Keywords",
    "ID": "Index Keywords",
    "PY": "Year",
    "TC": "Times Cited",
}

MULTIVALUED_COLS = ["AU", "DE", "ID", "RI", "C1", "AU_CO", "AU_IN", "AU_C1"]

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
    ...         "AU": "author 0;author 1;author 2,author 3,author 4".split(","),
    ...         "RecID": list(range(3)),
    ...      }
    ... )
    >>> x
                               AU  RecID
    0  author 0;author 1;author 2      0
    1                    author 3      1
    2                    author 4      2

    >>> __explode(x, 'AU')
             AU  RecID
    0  author 0      0
    1  author 1      0
    2  author 2      0
    3  author 3      1
    4  author 4      2

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
    with tqdm(total=8) as pbar:
        #
        # 1. Rename and seleect columns
        #
        x = x.copy()
        x = x.rename(columns=SCOPUS_TO_WOS_NAMES)
        x = x[[w for w in x.columns if w in SCOPUS_TO_WOS_NAMES.values()]]
        pbar.update(1)
        #
        # 2. Change ',' by ';' and remove '.' in author names
        #
        x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)
        if "AU" in x.columns:
            x["AU"] = x.AU.map(
                lambda w: w.replace(",", ";").replace(".", "")
                if pd.isna(w) is False
                else w
            )
        pbar.update(1)
        #
        # 3. Remove part of title in foreign language
        #
        if "TI" in x.columns:
            x["TI"] = x.TI.map(
                lambda w: w[0 : w.find("[")]
                if pd.isna(w) is False and w[-1] == "]"
                else w
            )
        pbar.update(1)
        #
        # 4. Keywords fusion
        #
        if "DE" in x.columns.tolist() and "ID" in x.columns.tolist():
            author_keywords = x["DE"].map(
                lambda x: x.split(";") if pd.isna(x) is False else []
            )
            index_keywords = x["ID"].map(
                lambda x: x.split(";") if pd.isna(x) is False else []
            )
            keywords = author_keywords + index_keywords
            keywords = keywords.map(lambda w: [e for e in w if e != ""])
            keywords = keywords.map(lambda w: [e.strip() for e in w])
            keywords = keywords.map(lambda w: sorted(set(w)))
            keywords = keywords.map(lambda w: ";".join(w))
            keywords = keywords.map(lambda w: None if w == "" else w)
            x["DE_ID"] = keywords
        pbar.update(1)
        #
        # 5. Extract country and affiliation
        #
        if "C1" in x.columns:

            x["AU_CO"] = x.C1.map(
                lambda w: extract_country(w) if pd.isna(w) is False else w
            )
            x["AU_IN"] = x.C1.map(
                lambda w: extract_institution(w) if pd.isna(w) is False else w
            )
        pbar.update(1)
        #
        # 6. Country and institution of first author
        #
        if "AU_CO" in x.columns:
            x["AU_CO1"] = x["AU_CO"].map(
                lambda w: w.split(";")[0] if not pd.isna(w) else w
            )
            x["AU_IN1"] = x["AU_IN"].map(
                lambda w: w.split(";")[0] if not pd.isna(w) else w
            )
        pbar.update(1)
        #
        # 7. Adds RecID
        #
        x["RecID"] = range(len(x))
        pbar.update(1)
        #
        # 8. Number of authors per document
        #
        x["N_AU"] = x["AU"].map(lambda w: len(w.split(";")) if not pd.isna(w) else 0)
        pbar.update(1)
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
    0         Authors                6      100.00%
    1    Author(s) ID                6      100.00%
    2  Index Keywords                6      100.00%
    3    Source title                5       83.33%
    4      None field                0        0.00%

    """

    return pd.DataFrame(
        {
            "Column": [
                ABBR_TO_NAMES[w] if w in ABBR_TO_NAMES else w for w in x.columns
            ],
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
    >>> x = pd.DataFrame({'AU': ['xxx', 'xxx; zzz', 'yyy', 'xxx; yyy; zzz']})
    >>> x
                  AU
    0            xxx
    1       xxx; zzz
    2            yyy
    3  xxx; yyy; zzz

    >>> extract_terms(x, column='AU')
        AU
    0  xxx
    1  yyy
    2  zzz

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

    >>> x = pd.DataFrame({'AU': ['xxx', 'xxx; zzz', 'yyy', 'xxx; yyy; zzz']})
    >>> x
                  AU
    0            xxx
    1       xxx; zzz
    2            yyy
    3  xxx; yyy; zzz

    >>> count_terms(x, column='AU')
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
    ...          'AU':  'xxx;xxx, zzz;yyy, xxx, yyy, zzz'.split(','),
    ...          'RI': '0;1,    3;4;,  4;,  5;,  6;'.split(','),
    ...          'SO': ' s0,     s0,   s1,  s1, s2'.split(','),
    ...          'DE': 'k0;k1, k0;k2, k3;k2;k1, k4, k5'.split(','),
    ...          'ID': 'w0;w1, w0;w2, w3;k1;w1, w4, w5'.split(','),
    ...          'PY': [1990, 1991, 1992, 1993, 1994],
    ...          'TC': list(range(5)),
    ...          'N_AU': [2, 2, 1, 1, 1],
    ...     }
    ... )
    >>> x # doctest: +NORMALIZE_WHITESPACE
             AU        RI       SO         DE         ID    PY  TC  N_AU
    0   xxx;xxx       0;1       s0      k0;k1      w0;w1  1990   0     2
    1   zzz;yyy      3;4;       s0      k0;k2      w0;w2  1991   1     2
    2       xxx        4;       s1   k3;k2;k1   w3;k1;w1  1992   2     1
    3       yyy        5;       s1         k4         w4  1993   3     1
    4       zzz        6;       s2         k5         w5  1994   4     1

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
    y["Years"] = str(min(x.PY)) + "-" + str(max(x.PY))
    y["Average citations per article"] = "{:4.2f}".format(x["TC"].mean())
    y["Authors"] = count_terms(x, "AU")
    y["Author(s) ID"] = count_terms(x, "RI")
    y["Articles per author"] = round(len(x) / count_terms(x, "AU"), 2)
    y["Authors per article"] = round(count_terms(x, "AU") / len(x), 2)
    y["Author Keywords"] = count_terms(x, "DE")
    y["Index Keywords"] = count_terms(x, "ID")
    y["Source titles"] = count_terms(x, "SO")

    y["Authors of single authored articles"] = len(x[x["N_AU"] == 1])
    y["Authors of multi authored articles"] = len(x[x["N_AU"] > 1])
    y["Co-authors per article"] = round(x["N_AU"].mean(), 2)
    y["Average articles per Source title"] = round(len(x) / count_terms(x, "SO"))
    # CAGR
    n = max(x.PY) - min(x.PY) + 1
    Po = len(x.PY[x.PY == min(x.PY)])
    Pn = len(x.PY[x.PY == max(x.PY)])
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
    ...          "PY": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "TC": list(range(10,16)),
    ...          "RecID": list(range(6)),
    ...     }
    ... )
    >>> df
         PY  TC  RecID
    0  2010  10      0
    1  2010  11      1
    2  2011  12      2
    3  2011  13      3
    4  2012  14      4
    5  2016  15      5

    >>> summary_by_year(df)[['Year', 'Times Cited', 'Num Documents', 'RecID']]
       Year  Times Cited  Num Documents   RecID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]

    >>> summary_by_year(df)[['Num Documents (Cum)', 'Times Cited (Cum)', 'Avg. Times Cited']]
       Num Documents (Cum)  Times Cited (Cum)  Avg. Times Cited
    0                    2                 21              10.5
    1                    4                 46              12.5
    2                    5                 60              14.0
    3                    5                 60               0.0
    4                    5                 60               0.0
    5                    5                 60               0.0
    6                    6                 75              15.0

    """
    data = df[["PY", "TC", "RecID"]].explode("PY")
    data["Num Documents"] = 1
    result = data.groupby("PY", as_index=False).agg(
        {"TC": np.sum, "Num Documents": np.size}
    )
    result = result.assign(
        RecID=data.groupby("PY").agg({"RecID": list}).reset_index()["RecID"]
    )
    result["TC"] = result["TC"].map(lambda x: int(x))
    years = [year for year in range(result.PY.min(), result.PY.max() + 1)]
    result = result.set_index("PY")
    result = result.reindex(years, fill_value=0)
    result["RecID"] = result["RecID"].map(lambda x: [] if x == 0 else x)
    result.sort_values(
        "PY", ascending=True, inplace=True,
    )
    result["Num Documents (Cum)"] = result["Num Documents"].cumsum()
    result["Times Cited (Cum)"] = result["TC"].cumsum()
    result["Avg. Times Cited"] = result["TC"] / result["Num Documents"]
    result["Avg. Times Cited"] = result["Avg. Times Cited"].map(
        lambda x: 0 if pd.isna(x) else x
    )
    result = result.reset_index()
    result = result.rename(columns=ABBR_TO_NAMES)
    return result


##
##
##  Analysis by term
##
##


def summarize_by_term(x, column, keywords=None):
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
    ...          "AU": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "TC": list(range(10,14)),
    ...          "RecID": list(range(4)),
    ...     }
    ... )
    >>> x
                               AU  TC  RecID
    0  author 0;author 1;author 2  10      0
    1                    author 0  11      1
    2                    author 1  12      2
    3                    author 3  13      3

    >>> summarize_by_term(x, 'AU')
        Authors  Num Documents  Times Cited   RecID
    0  author 0              2           21  [0, 1]
    1  author 1              2           22  [0, 2]
    2  author 2              1           10     [0]
    3  author 3              1           13     [3]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> summarize_by_term(x, 'AU', keywords=keywords)
        Authors  Num Documents  Times Cited   RecID
    0  author 1              2           22  [0, 2]
    1  author 2              1           10     [0]

    """
    x = __explode(x[[column, "TC", "RecID"]], column)
    x["Num Documents"] = 1
    result = x.groupby(column, as_index=False).agg(
        {"Num Documents": np.size, "TC": np.sum}
    )
    result = result.assign(
        RecID=x.groupby(column).agg({"RecID": list}).reset_index()["RecID"]
    )
    result["TC"] = result["TC"].map(lambda x: int(x))
    if keywords is not None:
        result = result[result[column].map(lambda w: w in keywords)]
    result.sort_values(
        [column, "Num Documents", "TC"],
        ascending=[True, False, False],
        inplace=True,
        ignore_index=True,
    )
    result = result.rename(columns=ABBR_TO_NAMES)
    return result


def documents_by_term(x, column, sep=None, keywords=None):
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
    ...          "AU": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "TC": list(range(10,14)),
    ...          "RecID": list(range(4)),
    ...     }
    ... )
    >>> x
                               AU  TC  RecID
    0  author 0;author 1;author 2  10      0
    1                    author 0  11      1
    2                    author 1  12      2
    3                    author 3  13      3

    >>> documents_by_term(x, 'AU')
        Authors  Num Documents   RecID
    0  author 0              2  [0, 1]
    1  author 1              2  [0, 2]
    2  author 2              1     [0]
    3  author 3              1     [3]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> documents_by_term(x, 'AU', keywords=keywords)
        Authors  Num Documents   RecID
    0  author 1              2  [0, 2]
    1  author 2              1     [0]

    """

    result = summarize_by_term(x, column, keywords)
    column = ABBR_TO_NAMES[column]
    result.pop("Times Cited")
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
    ...          "AU": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "TC": list(range(10,14)),
    ...          "RecID": list(range(4)),
    ...     }
    ... )
    >>> x
                               AU  TC  RecID
    0  author 0;author 1;author 2  10      0
    1                    author 0  11      1
    2                    author 1  12      2
    3                    author 3  13      3

    >>> citations_by_term(x, 'AU')
        Authors  Times Cited   RecID
    0  author 1           22  [0, 2]
    1  author 0           21  [0, 1]
    2  author 3           13     [3]
    3  author 2           10     [0]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> citations_by_term(x, 'AU', keywords=keywords)
        Authors  Times Cited   RecID
    0  author 1           22  [0, 2]
    1  author 2           10     [0]


    """
    result = summarize_by_term(x, column, keywords)
    column = ABBR_TO_NAMES[column]
    result.pop("Num Documents")
    result.sort_values(
        ["Times Cited", column],
        ascending=[False, True],
        inplace=True,
        ignore_index=True,
    )
    return result


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
