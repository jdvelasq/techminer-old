"""
Data Importation and Manipulation Functions
==================================================================================================



"""

import json
import logging
import re
import string
from os.path import dirname, join

import ipywidgets as widgets
import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords

from techminer.params import EXCLUDE_COLS, MULTIVALUED_COLS
from techminer.text import remove_accents
from techminer.thesaurus import text_clustering

#  from IPython.display import HTML, clear_output, display
# from ipywidgets import AppLayout, Layout
#  from techminer.explode import MULTIVALUED_COLS, __explode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


#  nltk.download("stopwords")

##
##
##  Data importation
##
##


def __disambiguate_authors(x):
    """Verify if author's names are unique. For duplicated names, based on `RI` column,
    adds a consecutive number to the name.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...   {
    ...      "Authors": "author 0;author 0;author 0,author 0,author 0".split(","),
    ...      "Authors_ID": "0;1;2;,3;,4;".split(','),
    ...   }
    ... )
    >>> df
                          Authors Authors_ID
    0  author 0;author 0;author 0     0;1;2;
    1                    author 0         3;
    2                    author 0         4;

    >>> __disambiguate_authors(df)
                                Authors Authors_ID
    0  author 0;author 0(1);author 0(2)      0;1;2
    1                       author 0(3)          3
    2                       author 0(4)          4

    """

    x["Authors"] = x.Authors.map(
        lambda w: w[:-1] if not pd.isna(w) and w[-1] == ";" else w
    )

    x["Authors_ID"] = x.Authors_ID.map(
        lambda w: w[:-1] if not pd.isna(w) and w[-1] == ";" else w
    )

    data = x[["Authors", "Authors_ID"]]
    data = data.dropna()

    data["*info*"] = [(a, b) for (a, b) in zip(data.Authors, data.Authors_ID)]

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

    x["Authors"] = x.Authors_ID.map(
        lambda z: ";".join([ids_names[w.strip()] for w in z.split(";")])
        if not pd.isna(z)
        else z
    )

    return x


def __MAP(x, column, f):
    """Applies function f to column in dataframe x.

    >>> import pandas as pd
    >>> x = pd.DataFrame({'Affiliations': ['USA; Russian Federation']})
    >>> __MAP(x, 'Affiliations', lambda w: __extract_country(w))
    0    United States;Russia
    Name: Affiliations, dtype: object


    """
    x = x.copy()
    if column in MULTIVALUED_COLS:
        z = x[column].map(lambda w: w.split(";") if not pd.isna(w) else w)
        z = z.map(lambda w: [f(z) for z in w] if isinstance(w, list) else w)
        z = z.map(
            lambda w: [z for z in w if not pd.isna(z)] if isinstance(w, list) else w
        )
        z = z.map(lambda w: ";".join(w) if isinstance(w, list) else w)
        return z
    return x[column].map(lambda w: f(w))


def __extract_country(x):
    """Extracts country name from a string,

    Examples
    ----------------------------------------------------------------------------------------------

    >>> __extract_country('United States of America')
    'United States'

    >>> __extract_country('USA')
    'United States'

    >>> __extract_country('Peoples R China')
    'China'

    >>> __extract_country('Russian Federation')
    'Russia'

    >>> __extract_country('xxx')
    <NA>

    >>> __extract_country('Department of Architecture, National University of Singapore, Singapore')
    'Singapore'

    """
    if pd.isna(x) or x is None:
        return pd.NA
    #
    # lista generica de nombres de paises
    #
    module_path = dirname(__file__)
    with open(join(module_path, "data/worldmap.data"), "r") as f:
        countries = json.load(f)
    country_names = list(countries.keys())

    # paises faltantes
    country_names.append("Singapore")
    country_names.append("Malta")
    country_names.append("United States")
    country_names = {country.lower(): country for country in country_names}

    # Reemplazo de nombres de regiones administrativas
    # por nombres de paises
    x = x.lower()
    x = x.strip()
    x = re.sub("united states of america", "united states", x)
    x = re.sub("usa", "united states", x)
    x = re.sub("bosnia and herzegovina", "bosnia and herz.", x)
    x = re.sub("czech republic", "czechia", x)
    x = re.sub("russian federation", "russia", x)
    x = re.sub("peoples r china", "china", x)
    x = re.sub("hong kong", "china", x)
    x = re.sub("macau", "china", x)
    x = re.sub("macao", "china", x)
    #
    for z in reversed(x.split(",")):
        z = z.strip()
        if z.lower() in country_names.keys():
            return country_names[z.lower()]
        # descarta espaciado
        z = z.lower()
        z = z.split(" ")
        z = [w.strip() for w in z]
        z = " ".join(z)
        if z.lower() in country_names.keys():
            return country_names[z.lower()]
    return pd.NA


def __extract_institution(x):
    """
    """

    def search_name(affiliation):
        if len(affiliation.split(",")) == 1:
            return x.strip()
        affiliation = affiliation.lower()
        for elem in affiliation.split(","):
            for name in names:
                if name in elem:
                    return elem.strip()
        return pd.NA

    #
    names = [
        "univ",
        "institut",
        "centre",
        "center",
        "centro",
        "agency",
        "council",
        "commission",
        "college",
        "politec",
        "inc.",
        "ltd.",
        "office",
        "department",
        "direction" "laboratory",
        "laboratoire",
        "colegio",
        "school",
        "scuola",
        "ecole",
        "hospital",
        "association",
        "asociacion",
        "company",
        "organization",
        "academy",
    ]

    if pd.isna(x) is True or x is None:
        return pd.NA
    institution = search_name(x)
    if pd.isna(institution):
        if len(x.split(",")) == 2:
            return x.split(",")[0].strip()
    return institution


def __NLP(text):
    """Extracts words  and 2-grams from phrases.
    """

    def obtain_n_grams(x, n):
        from nltk.corpus import stopwords

        stopwords = stopwords.words("english")
        if len(x.split()) < n + 1:
            return [x.strip()]
        words = x.split()
        text = [words[index : index + n] for index in range(len(words) - n + 1)]
        text = [
            phrase for phrase in text if not any([word in stopwords for word in phrase])
        ]
        text = [[word for word in phrase if not word.isdigit()] for phrase in text]
        lem = WordNetLemmatizer()
        text = [[lem.lemmatize(word) for word in phrase] for phrase in text]
        text = [[word for word in phrase if word != ""] for phrase in text]
        text = [" ".join(phrase) for phrase in text if len(phrase)]
        return text

    from nltk.corpus import stopwords

    stopwords = stopwords.words("english")

    nlp_result = []

    text = text.lower()
    text = text.split(".")
    text = [word for phrase in text for word in phrase.split(";")]
    text = [word for phrase in text for word in phrase.split(",")]
    text = [
        phrase.translate(str.maketrans("", "", string.punctuation)) for phrase in text
    ]

    two_grams = [obtain_n_grams(phrase, 2) for phrase in text]
    two_grams = [word for phrase in two_grams for word in phrase]
    two_grams = [phrase for phrase in two_grams if phrase != ""]

    three_grams = [obtain_n_grams(phrase, 3) for phrase in text]
    three_grams = [word for phrase in three_grams for word in phrase]
    three_grams = [phrase for phrase in three_grams if phrase != ""]
    three_grams = sorted(set(three_grams))

    text = [word for phrase in text for word in phrase.split() if not word.isdigit()]
    text = [word for word in text if word not in stopwords]
    text = [word for word in text if word != ""]

    ###
    text = sorted(set(text + two_grams + three_grams))
    text = [w.split() for w in text]
    text = [nltk.pos_tag(w) for w in text]
    tags = [" ".join([t for _, t in v]) for v in text]
    text = [" ".join([t for t, _ in v]) for v in text]
    text = [
        t
        for k, t in zip(tags, text)
        if k in ["NN", "JJ NN", "NN NN", "NNS", "NN NNS", "JJ NNS", "NN NNS",]
    ]
    ###

    # lem = WordNetLemmatizer()
    # text = sorted(set([lem.lemmatize(word) for word in text]))
    return ";".join(sorted(set(text)))


def load_scopus(x):
    """Import filter for Scopus data.
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    scopus2tags = {
        "Abbreviated Source Title": "Abb_Source_Title",
        "Abstract": "Abstract",
        "Access Type": "Access_Type",
        "Affiliations": "Affiliations",
        "Art. No.": "Art_No",
        "Author Keywords": "Author_Keywords",
        "Author(s) ID": "Authors_ID",
        "Authors with affiliations": "Authors_with_affiliations",
        "Authors": "Authors",
        "Cited by": "Times_Cited",
        "CODEN": "CODEN",
        "Correspondence Address": "Correspondence_Address",
        "Document Type": "Document_Type",
        "DOI": "DOI",
        "Editors": "Editors",
        "EID": "EID",
        "Index Keywords": "Index_Keywords",
        "ISBN": "ISBN",
        "ISSN": "ISSN",
        "Issue": "Issue",
        "Language of Original Document": "Language_of_Original_Document",
        "Link": "Link",
        "Page count": "Page_count",
        "Page end": "Page_end",
        "Page start": "Page_start",
        "Publication Stage": "Publication_Stage",
        "Publisher": "Publisher",
        "PubMed ID": "PubMed_ID",
        "Source title": "Source_title",
        "Source": "Source",
        "Title": "Title",
        "Volume": "Volume",
        "Year": "Year",
    }

    x = x.copy()
    logging.info("Renaming and selecting columns ...")
    x = x.rename(columns=scopus2tags)

    logging.info("Removing accents ...")
    x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)

    if "Authors" in x.columns:

        logging.info('Removing  "[No author name available]" ...')
        x["Authors"] = x.Authors.map(
            lambda w: pd.NA if w == "[No author name available]" else w
        )

        logging.info("Formatting author names ...")
        x["Authors"] = x.Authors.map(
            lambda w: w.replace(",", ";").replace(".", "") if pd.isna(w) is False else w
        )

        logging.info("Counting number of authors per document...")
        x["Num_Authors"] = x.Authors.map(
            lambda w: len(w.split(";")) if not pd.isna(w) else 0
        )

    if "Authors_ID" in x.columns:
        x["Authors_ID"] = x.Authors_ID.map(
            lambda w: pd.NA if w == "[No author id available]" else w
        )

    if "Authors" in x.columns and "Authors_ID" in x.columns:
        logging.info("Disambiguate author names ...")
        x = __disambiguate_authors(x)

    if "Title" in x.columns:
        logging.info("Removing part of titles in foreing languages ...")
        x["Title"] = x.Title.map(
            lambda w: w[0 : w.find("[")] if pd.isna(w) is False and w[-1] == "]" else w
        )

    if "Affiliations" in x.columns:
        logging.info("Extracting country names ...")
        x["Countries"] = __MAP(x, "Affiliations", __extract_country)

        logging.info("Extracting country of first author ...")
        x["Country_1st_Author"] = x.Countries.map(
            lambda w: w.split(";")[0] if isinstance(w, str) else w
        )

        logging.info("Extracting institutions from affiliations ...")
        x["Institutions"] = __MAP(x, "Affiliations", __extract_institution)

        logging.info("Extracting institution of first author ...")
        x["Institution_1st_Author"] = x.Institutions.map(
            lambda w: w.split(";")[0] if isinstance(w, str) else w
        )

        logging.info("Reducing list of countries ...")
        x["Countries"] = x.Countries.map(
            lambda w: ";".join(set(w.split(";"))) if isinstance(w, str) else w
        )

    if "Author_Keywords" in x.columns:
        logging.info("Transforming Author Keywords to lower case ...")
        x["Author_Keywords"] = x.Author_Keywords.map(
            lambda w: w.lower() if not pd.isna(w) else w
        )
        x["Author_Keywords"] = x.Author_Keywords.map(
            lambda w: ";".join([z.strip() for z in w.split(";")])
            if not pd.isna(w)
            else w
        )

    if "Index_Keywords" in x.columns:
        logging.info("Transforming Index Keywords to lower case ...")
        x["Index_Keywords"] = x.Index_Keywords.map(
            lambda w: w.lower() if not pd.isna(w) else w
        )
        x["Index_Keywords"] = x.Index_Keywords.map(
            lambda w: ";".join([z.strip() for z in w.split(";")])
            if not pd.isna(w)
            else w
        )

    if "Abstract" in x.columns:

        x.Abstract = x.Abstract.map(
            lambda w: w[0 : w.find("\u00a9")] if not pd.isna(w) else w
        )

        logging.info("Extracting Abstract words ...")
        x["Abstract_words"] = __MAP(x, "Abstract", __NLP)
        # logging.info("Clustering Abstract Keywords ...")
        #  thesaurus = text_clustering(x.Abstract_CL, transformer=lambda u: u.lower())
        #  logging.info("Cleaning Abstract Keywords ...")
        #  thesaurus = thesaurus.compile()
        # x["Abstract_CL"] = __MAP(x, "Abstract_CL", thesaurus.apply)

    if "Title" in x.columns:
        logging.info("Extracting Title words ...")
        x["Title_words"] = __MAP(x, "Title", __NLP)
        # logging.info("Clustering Title Keywords ...")
        #  thesaurus = text_clustering(x.Title_CL, transformer=lambda u: u.lower())
        #  logging.info("Cleaning Title Keywords ...")
        #  thesaurus = thesaurus.compile()
        #  x["Title_CL"] = __MAP(x, "Title_CL", thesaurus.apply)

    if "Times_Cited" in x.columns:
        logging.info("Removing <NA> from Times_Cited field ...")
        x["Times_Cited"] = x["Times_Cited"].map(lambda w: 0 if pd.isna(w) else w)

    if "Abb_Source_Title" in x.columns:
        logging.info("Removing '.' from Abb_Source_Title field ...")
        x["Abb_Source_Title"] = x["Abb_Source_Title"].map(
            lambda w: w.replace(".", "") if isinstance(w, str) else w
        )

    x["ID"] = range(len(x))

    x = x.applymap(lambda w: pd.NA if isinstance(w, str) and w == "" else w)

    logging.getLogger().disabled = True

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
    x[column] = x[column].map(
        lambda w: w.split(";") if not pd.isna(w) and isinstance(w, str) else w
    )
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
            y[column] = count_terms(x, column)
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
            y["Documents per author"] = round(len(x) / count_terms(x, "Authors"), 2)
            y["Authors per document"] = round(count_terms(x, "Authors") / len(x), 2)
        if "Num_Authors" in x.columns:
            y["Authors of single authored documents"] = len(x[x["Num_Authors"] == 1])
            y["Authors of multi authored documents"] = len(x[x["Num_Authors"] > 1])
            y["Co-authors per document"] = round(x["Num_Authors"].mean(), 2)
        if "Source_Title" in x.columns:
            y["Average documents per Source title"] = round(
                len(x) / count_terms(x, "Source_title")
            )
    #
    d = [key for key in y.keys()]
    v = [y[key] for key in y.keys()]
    return pd.DataFrame(v, columns=["value"], index=d)


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


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
