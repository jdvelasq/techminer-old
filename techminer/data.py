
"""
Data Importation and Manipulation Functions
==================================================================================================



"""

import logging

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.text import extract_country, extract_institution, remove_accents
from techminer.thesaurus import text_clustering

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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

NORMALIZED_NAMES = {
    "AB": "Abstract",
    "Abbreviated Source Title": "_Abbreviated Source Title",
    "Access Type": "_Access Type",
    "Affiliations": "Affiliations",
    "AR": "_Art. No.",
    "Art. No.": "_Art. No.",
    "AU": "Authors",
    "Author Keywords": "Author Keywords",
    "Author(s) ID": "Author(s) ID",
    "Authors with affiliations": "_Authors with affiliations",
    "Authors": "Authors",
    "BE": "Editors",
    "BN": "_ISBN",
    "BP": "_Page start",
    "C1": "Affiliations",
    "Chemicals/CAS": "_Chemicals/CAS",
    "Cited by": "Cited by",
    "CODEN": "_CODEN",
    "Conference code": "_Conference code",
    "Conference date": "_Conference date",
    "Conference location": "_Conference location",
    "Conference name": "_Conference name",
    "Correspondence Address": "_Correspondence Address",
    "CR": "Cited references",
    "DE": "Author Keywords",
    "Document Type": "_Document Type",
    "DOI": "_DOI",
    "DT": "_Document Type",
    "Editors": "_Editors",
    "EID": "_EID",
    "EP": "_Page end",
    "FN": "_Source",
    "Funding Details": "_Funding Details",
    "Funding Text 1": "_Funding Text 1",
    "Index Keywords": "Index Keywords",
    "IS": "_Issue",
    "ISBN": "_ISBN",
    "ISSN": "_ISSN",
    "Issue": "_Issue",
    "J9": "_Abbreviated Source Title",
    "LA": "_Language of Original Document",
    "Language of Original Document": "_Language of Original Document",
    "Link": "_Link",
    "Manufacturers": "_Manufacturers",
    "Molecular Sequence Numbers": "_Molecular Sequence Numbers",
    "OA": "_Access Type",
    "Page count": "_Page count",
    "Page end": "_Page end",
    "Page start": "_Page start",
    "PG": "_Page count",
    "PM": "_PubMed ID",
    "PT": "Year",
    "PU": "_Publisher",
    "Publication Stage": "_Publication Stage",
    "Publisher": "_Publisher",
    "PubMed ID": "_PubMed ID",
    "References": "References",
    "RI": "Author(s) ID",
    "SC": "_Subject",
    "SN": "_ISSN",
    "SO": "Source title",
    "Source title": "Source title",
    "Source": "_Source",
    "Sponsors": "_Sponsors",
    "TC": "Cited by",
    "TI": "Title",
    "Title": "Title",
    "Tradenames": "_Tradenames",
    "UT": "_EID",
    "VL": "_Volume",
    "Volume": "_Volume",
    "Year": "Year",
}


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
    keywords = []
    if "Author Keywords" in x.columns.tolist():
        x['Author Keywords'] = x['Author Keywords'].map(lambda w: w.lower() if pd.isna(w) is False else w)
        keywords = x['Author Keywords'].tolist()
    if "Index Keywords" in x.columns.tolist():
        x['Index Keywords'] = x['Index Keywords'].map(lambda w: w.lower() if pd.isna(w) is False else w)
        keywords = x['Index Keywords'].tolist()
    if len(keywords) > 0:
        logging.info("Building thesaurus for author and index keywords ...")
        keywords = pd.Series(keywords)
        thesaurus = text_clustering(keywords, sep=';', transformer=lambda u: u.lower())
        thesaurus = thesaurus.compile()
        logging.info("Cleaning Author Keywords ...")
        x['Author Keywords (Cleaned)'] = x['Author Keywords'].map(
            lambda w: ';'.join([thesaurus.apply(z) for z in w.split(';')]) if pd.isna(w) is False else pd.NA
        )
        logging.info("Cleaning Index Keywords ...")
        x['Index Keywords (Cleaned)'] = x['Index Keywords'].map(
            lambda w: ';'.join([thesaurus.apply(z) for z in w.split(';')]) if pd.isna(w) is False else pd.NA
        )
    #
    if (
        "Author Keywords (Cleaned)" in x.columns.tolist()
        and "Index Keywords (Cleaned)" in x.columns.tolist()
    ):
        #
        logging.info("Fusioning cleaned author and index keywords ...")
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
        x["Country 1st"] = x["Countries"].map(
            lambda w: w.split(";")[0] if not pd.isna(w) else w
        )
        x["Countries"] = x["Countries"].map(lambda w: ';'.join(sorted(set(w.split(';')))) if pd.isna(w) is False else pd.NA)

    #
    if "Affiliations" in x.columns:
        #
        logging.info("Extracting institutions from affiliations ...")
        #
        x["Institutions"] = x.Affiliations.map(
            lambda w: extract_institution(w) if pd.isna(w) is False else w
        )
        x["Institution 1st"] = x["Institutions"].map(
            lambda w: w.split(";")[0] if not pd.isna(w) else w
        )
        x["Institutions"] = x["Institutions"].map(lambda w: ';'.join(sorted(set(w.split(';')))) if pd.isna(w) is False else pd.NA)        
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
    x = x.copy()
    x[column] = x[column].map(lambda w: w.split(";") if not pd.isna(w) and isinstance(w, str) else w)
    x = x.explode(column)
    x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
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
    Authors                                      3
    Articles per author                       1.67
    Authors per article                        0.6
    Authors of single authored articles          3
    Authors of multi authored articles           2
    Co-authors per article                     1.4
    Author(s) ID                                 7
    Source title                                 3
    Author Keywords                              6
    Index Keywords                               7
    Years                                1990-1994
    Compound annual growth rate              0.0 %
    Cited by                                     5
    Average citations per article             2.00
    Num Authors                                  2
    
    """
    y = {}
    y["Articles"] = str(len(x))
    #
    for column in x.columns:
        if column[0] == '_':
            continue
        if column != 'Year':
            y[column] = count_terms(x, column)
        if column == "Year":
            y["Years"] = str(min(x.Year)) + "-" + str(max(x.Year))
            n = max(x.Year) - min(x.Year) + 1
            Po = len(x.Year[x.Year == min(x.Year)])
            Pn = len(x.Year[x.Year == max(x.Year)])
            cagr = str(round(100 * (np.power(Pn / Po, 1 / n) - 1), 2)) + " %"
            y["Compound annual growth rate"] = cagr
        if  column == "Cited by":
            y["Average citations per article"] = "{:4.2f}".format(x["Cited by"].mean())
        if column == "Authors":
            y["Articles per author"] = round(len(x) / count_terms(x, "Authors"), 2)
            y["Authors per article"] = round(count_terms(x, "Authors") / len(x), 2)
        if "Num Authors" in x.columns:
            y["Authors of single authored articles"] = len(x[x["Num Authors"] == 1])
            y["Authors of multi authored articles"] = len(x[x["Num Authors"] > 1])
            y["Co-authors per article"] = round(x["Num Authors"].mean(), 2)
        if "Source Title" in x.columns:
            y["Average articles per Source title"] = round(
                len(x) / count_terms(x, "Source title")
            )
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


#     def keywords_completation(self):
#         """Complete keywords in 'Keywords' column (if exists) from title and abstract.
#         """

#         cp = self.copy()
#         if "Keywords" not in self.columns:
#             cp = cp.keywords_fusion()

#         # Remove copyright character from abstract
#         abstract = cp.Abstract.map(
#             lambda x: x[0 : x.find("\u00a9")]
#             if isinstance(x, str) and x.find("\u00a9") != -1
#             else x
#         )

#         title_abstract = cp.Title + " " + abstract

#         kyw = Keywords()
#         kyw.add_keywords(cp.Keywords, sep=";")

#         keywords = title_abstract.map(lambda x: kyw.extract_from_text(x, sep=";"))
#         idx = cp.Keywords.map(lambda x: x is None)
#         cp.loc[idx, "Keywords"] = keywords[idx]

#         return DataFrame(cp)



##
##
##  APP
##
##
