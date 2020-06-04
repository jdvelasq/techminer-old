
"""
Scopus Data
==================================================================================================



"""

import logging

import pandas as pd
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

#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()