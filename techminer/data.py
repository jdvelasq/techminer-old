
"""
Data Importation and Manipulation Functions
==================================================================================================



"""

import json
import logging
import re
from os.path import dirname, join

import ipywidgets as widgets
import numpy as np
import pandas as pd

# from IPython.display import HTML, clear_output, display
# from ipywidgets import AppLayout, Layout
# from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.text import remove_accents

#  extract_country, extract_institution, 
from techminer.thesaurus import text_clustering

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


##
##  Augmented Field Tags (based on WoS convention):
##  

#
#  AB     Abstract
#  AE     Patent Assignee
#  AR     Article Number // Art. No.
#  AU     Authors // Authors
#  AF     Author Full Name
#  BA     Book Author(s)
#  BE     Editors \\ Editors
#  BF     Book Authors Full Name
#  BN     International Standard Book Number \\ ISBN
#  BP     Beginning Page // Page start
#  BS     Book Series Subtitle
#  C1     Author Address \\  Authors with affiliations
#  CA     Group Authors
#  CL     Conference Location
#  CT     Conference Title
#  CR     Cited References
#  CY     Conference Date
#  D2     Book Digital Object Identifier (DOI)
#  DA     Date this report was generated
#  DE     Author Keywords \\ Author Keywords
#  DI     Digital Object Identifier \\ DOI
#  DT     Document Type \\ Document Type
#  EF     End of File
#  EI     Electronic International Standard Serial Number (eISSN)
#  EM     E-mail Address
#  EP     Ending Page // Page end
#  ER     End of Record
#  FN     File Name \\ Source
#  FT     Foreign Title
#  FU     Funding Agency and Grant Numbe
#  FX     Funding Text
#  J9     29-Character Source Abbreviation \\ Abbreviated Source Title
#  JI     ISO Source Abbreviation
#  GA     Document Delivery Number
#  GP     Book Group Authors
#  HC     ESI Highly Cited Paper.
#  HP     ESI Hot Paper.
#  HO     Conference Host
#  ID     Keywords Plus \\ Index Keywords
#  IS     Issue
#  LA     Language \\ Language of Original Document
#  MA     Meeting Abstract Number
#  NR     Cited Reference Count
#  OA     Open Access Indicator
#  OI     ORCID Identifier
#  PA     Publisher Address
#  PG     Page count
#  PD     Publication Date
#  PI     Publisher City
#  PM     PubMed ID \\ PubMed ID
#  PN     Part / Patent Number
#  PT     Publication Type (J=Journal; B=Book; S=Series; P=Patent)
#  PY     Publication Year \\ Year
#  PU     Publisher \\ Publisher
#  P2     Chapter Count 
#  RI     ResearcherID Number \\ Author(s) ID
#  RP     Reprint Address
#  SC     Research Areas
#  SE     Book Series Title
#  SI     Special Issue
#  SN     International Standard Serial Number \\ ISSN
#  SO     Full Source Title \\ Source title
#  SP     Conference Sponsor
#  SU     Supplement
#  TC     Times Cited Count \\ Cited by
#  TI     Title \\ Title
#  U1     Usage Count (Last 180 Days)
#  U2     Usage Count (Since 2013)
#  UT     Accession Number / ISI Unique Article Identifier
#  VL     Volume
#  VR     Version Number
#  Z1     Title (in second language)
#  Z2     Author(s) (in second language)
#  Z3     Full Source Title (in second language) (includes title and subtitle)
#  Z4     Abstract (in second language)
#  Z8     Times Cited in Chinese Science Citation Database
#  Z9     Total Times Cited Count
#  ZB     Times Cited in BIOSIS Citation Index
#  WC     Web of Science Categories
#  SEID   Scopus EID
#  STY    Scopus Type
#  SPS    Scopus Publication Stage
#  SCD    Scopus CODEN
#  SLK    Scopus Link
#  SAF    Scopus Affiliations    
#  SCA    Scopus Correspondence Address
#  SA     Scopus Access Type
#
#  AUN    Number of authors per document
#  AUCO   Author's countries
#  AUCO1  Country of first author
#  AUIN   Author's Institutions 
#  AUIN1  Institution of first author
#

##
##
##  Data importation
##
##

# NORMALIZED_NAMES = {
#     "AB": "Abstract",
#     "Abbreviated Source Title": "*Abbreviated Source Title",
#     "Access Type": "*Access Type",
#     "Affiliations": "Affiliations",
#     "AR": "*Art. No.",
#     "Art. No.": "*Art. No.",
#     "AU": "Authors",
#     "Author Keywords": "Author Keywords",
#     "Author(s) ID": "Author(s) ID",
#     "Authors with affiliations": "*Authors with affiliations",
#     "Authors": "Authors",
#     "BE": "Editors",
#     "BN": "*ISBN",
#     "BP": "*Page start",
#     "C1": "Affiliations",
#     "Chemicals/CAS": "*Chemicals/CAS",
#     "Cited by": "Cited by",
#     "CODEN": "*CODEN",
#     "Conference code": "*Conference code",
#     "Conference date": "*Conference date",
#     "Conference location": "*Conference location",
#     "Conference name": "*Conference name",
#     "Correspondence Address": "*Correspondence Address",
#     "CR": "Cited references",
#     "DE": "Author Keywords",
#     "Document Type": "*Document Type",
#     "DOI": "*DOI",
#     "DT": "*Document Type",
#     "Editors": "*Editors",
#     "EID": "*EID",
#     "EP": "*Page end",
#     "FN": "*Source",
#     "Funding Details": "*Funding Details",
#     "Funding Text 1": "*Funding Text 1",
#     "Index Keywords": "Index Keywords",
#     "IS": "*Issue",
#     "ISBN": "*ISBN",
#     "ISSN": "*ISSN",
#     "Issue": "*Issue",
#     "J9": "*Abbreviated Source Title",
#     "LA": "*Language of Original Document",
#     "Language of Original Document": "*Language of Original Document",
#     "Link": "*Link",
#     "Manufacturers": "*Manufacturers",
#     "Molecular Sequence Numbers": "*Molecular Sequence Numbers",
#     "OA": "*Access Type",
#     "Page count": "*Page count",
#     "Page end": "*Page end",
#     "Page start": "*Page start",
#     "PG": "*Page count",
#     "PM": "*PubMed ID",
#     "PT": "Year",
#     "PU": "*Publisher",
#     "Publication Stage": "*Publication Stage",
#     "Publisher": "*Publisher",
#     "PubMed ID": "*PubMed ID",
#     "References": "References",
#     "RI": "Author(s) ID",
#     "SC": "*Subject",
#     "SN": "*ISSN",
#     "SO": "Source title",
#     "Source title": "Source title",
#     "Source": "*Source",
#     "Sponsors": "*Sponsors",
#     "TC": "Cited by",
#     "TI": "Title",
#     "Title": "Title",
#     "Tradenames": "*Tradenames",
#     "UT": "*EID",
#     "VL": "*Volume",
#     "Volume": "*Volume",
#     "Year": "Year",
# }

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
                              Authors     Authors_ID
    0  author 0;author 0;author 0  0;1;2;
    1                    author 0      3;
    2                    author 0      4;

    >>> __disambiguate_authors(df)
                                    Authors    Authors_ID
    0  author 0;author 0(1);author 0(2)  0;1;2
    1                       author 0(3)      3
    2                       author 0(4)      4

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
    """


    x = x.copy()
    if column[0] in [
        "Authors",
        "Authors_ID",
        "Author_Keywords",
        "Author_Keywords_ID",
        "Index_Keywords",
        "Index_Keywords_ID",
        "Institutions",
        "Countries",
    ]:
        z = x[column].map(lambda w: w.split(';') if not pd.isna(w) else w)
        z = z.map(lambda w: [f(z) for z in w] if isinstance(w, list) else w)
        z = z.map(lambda w: [z for z in w if not pd.isna(z)] if isinstance(w, list) else w)
        z = z.map(lambda w: ';'.join(w) if isinstance(w, list) else w)
        return z
    return x[column].map(lambda w: f(w))
    

def __extract_country(x):
    """Extracts country name from a string,

    Examples
    ----------------------------------------------------------------------------------------------

    >>> extract_country('United States of America')
    'United States'

    >>> extract_country('Peoples R China')
    'China'

    >>> extract_country('xxx')
    <NA>

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
    #
    # paises faltantes
    #
    country_names.append("Singapore")
    country_names.append("Malta")
    country_names.append("United States")
    #
    # Reemplazo de nombres de regiones administrativas
    # por nombres de paises
    #
    x = x.title()
    x = re.sub("United States Of America", "United States", x)
    x = re.sub("USA", "United States", x)
    x = re.sub("Bosnia and Herzegovina", "Bosnia and Herz.", x)
    x = re.sub("Czech Republic", "Czechia", x)
    x = re.sub("Russian Federation", "Russia", x)
    x = re.sub("Hong Kong", "China", x)
    x = re.sub("Macau", "China", x)
    x = re.sub("Macao", "China", x)
    #
    for z in x.split(','):
        z = z.split(' ')
        z = [w.strip() for w in z]
        z = ' '.join(z)
        for country in country_names:
            if country.title() in z:
                return country
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



def load_scopus(x):
    """Import filter for Scopus data.
    """

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
        "Cited by": "Cited_by",
        "CODEN": "CODEN",
        "Correspondence Address": "Correspondence_Address",
        "Document Type": "Document_Type",
        "DOI": "DOI",
        "Editors": "Editors",
        "EID": "EID",
        "Index Keywords": "Index_Keywords",
        "ISBN": "ISBN",
        "ISSN": "ISSN",
        "Issue": 'Issue',
        "Language of Original Document": "Language_of_Original_Document",
        "Link": "Link",
        "Page count": "Page_count",
        "Page end": "Page_end",
        "Page start": "Page_start",
        "Publication Stage": "Publication_Stage",
        "Publisher": "Publisher",
        "PubMed ID": "PubMed ID",
        "Source title": "Source title",
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
        x["Country_1st_Author"] = x.Countries.map(lambda w: w.split(';')[0] if isinstance(w, str) else w)

        logging.info("Extracting institutions from affiliations ...")
        x["Institutions"] = __MAP(x, "Affiliations", __extract_institution)
        
        logging.info("Extracting institution of first author ...")
        x["Institution_1st_Author"] = x.Institutions.map(lambda w: w.split(';')[0] if isinstance(w, str) else w)

    if "Author_Keywords" in x.columns:
        logging.info("Transforming Author Keywords to lower case ...")
        x['Author_Keywords'] = x.Author_Keywords.map(lambda w: w.lower() if not pd.isna(w) else w)

    if "Index_Keywords" in x.columns:
        logging.info("Transforming Index Keywords to lower case ...")
        x['Index_Keywords'] = x.Index_Keywords.map(lambda w: w.lower() if not pd.isna(w) else w)

    keywords = []
    if "Author_Keywords" in x.columns:
        keywords += x.Author_Keywords.tolist()
    if "Index_Keywords" in x.columns:
        keywords += x.Index_Keywords.tolist()
    if len(keywords) > 0:
        keywords = pd.Series(keywords)
        keywords = keywords.map(lambda w: w.split(';') if isinstance(w, str) else w)
        keywords = keywords.explode()
        keywords = keywords.reset_index(drop=True)
        logging.info("Clustering keywords ...")
        thesaurus = text_clustering(keywords, transformer=lambda u: u.lower())
        thesaurus = thesaurus.compile()
        if "Author_Keywords" in x.columns:
            logging.info("Cleaning Author Keywords ...")
            x["Author_Keywords_CL"] = __MAP(x, "Author_Keywords", thesaurus.apply)
            x["Author_Keywords_CL"] = __MAP(x, "Author_Keywords_CL", lambda w: w.strip() if not pd.isna(w) else w)
        if "Index_Keywords" in x.columns:
            logging.info("Cleaning Index Keywords ...")
            x["Index_Keywords_CL"] = __MAP(x, "Index_Keywords", thesaurus.apply)
            x["Index_Keywords_CL"] = __MAP(x, "Index_Keywords_CL", lambda w: w.strip() if not pd.isna(w) else w)    
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
        if column[0] == '*':
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
