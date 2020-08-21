"""

"""

import json
import textwrap
import re
import string
from os.path import dirname, join

import nltk
import pandas as pd
from nltk import WordNetLemmatizer
import datetime


from techminer.core.extract_words import extract_words
from techminer.core.text import remove_accents
from techminer.core.map import map_

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


def import_scopus(input_file="scopus.csv", output_file="techminer.csv"):
    #
    def logging_info(msg):
        print(
            "{} - INFO - {}".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), msg
            )
        )

    #

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
        "References": "Global_References",
        "Source title": "Source_title",
        "Source": "Source",
        "Title": "Title",
        "Volume": "Volume",
        "Year": "Year",
    }

    x = pd.read_csv(input_file)
    logging_info("Renaming and selecting columns ...")
    x = x.rename(columns=scopus2tags)

    logging_info("Removing accents ...")
    x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)

    if "Authors" in x.columns:

        logging_info('Removing  "[No author name available]" ...')
        x["Authors"] = x.Authors.map(
            lambda w: pd.NA if w == "[No author name available]" else w
        )

        logging_info("Formatting author names ...")
        x["Authors"] = x.Authors.map(
            lambda w: w.replace(",", ";").replace(".", "") if pd.isna(w) is False else w
        )

        logging_info("Counting number of authors per document...")
        x["Num_Authors"] = x.Authors.map(
            lambda w: len(w.split(";")) if not pd.isna(w) else 0
        )

        logging_info("Counting frac number of documents per author...")
        x["Frac_Num_Documents"] = x.Authors.map(
            lambda w: 1.0 / len(w.split(";")) if not pd.isna(w) else 0
        )

    if "Authors_ID" in x.columns:
        x["Authors_ID"] = x.Authors_ID.map(
            lambda w: pd.NA if w == "[No author id available]" else w
        )

    if "Authors" in x.columns and "Authors_ID" in x.columns:
        logging_info("Disambiguate author names ...")
        x = __disambiguate_authors(x)

    if "Title" in x.columns:
        logging_info("Removing part of titles in foreing languages ...")
        x["Title"] = x.Title.map(
            lambda w: w[0 : w.find("[")] if pd.isna(w) is False and w[-1] == "]" else w
        )

    if "Affiliations" in x.columns:
        logging_info("Extracting country names ...")
        x["Countries"] = map_(x, "Affiliations", __extract_country)

        logging_info("Extracting country of first author ...")
        x["Country_1st_Author"] = x.Countries.map(
            lambda w: w.split(";")[0] if isinstance(w, str) else w
        )

        logging_info("Extracting institutions from affiliations ...")
        x["Institutions"] = map_(x, "Affiliations", __extract_institution)

        logging_info("Extracting institution of first author ...")
        x["Institution_1st_Author"] = x.Institutions.map(
            lambda w: w.split(";")[0] if isinstance(w, str) else w
        )

        logging_info("Reducing list of countries ...")
        x["Countries"] = x.Countries.map(
            lambda w: ";".join(set(w.split(";"))) if isinstance(w, str) else w
        )

    if "Author_Keywords" in x.columns:
        logging_info("Transforming Author Keywords to lower case ...")
        x["Author_Keywords"] = x.Author_Keywords.map(
            lambda w: w.lower() if not pd.isna(w) else w
        )
        x["Author_Keywords"] = x.Author_Keywords.map(
            lambda w: ";".join(sorted([z.strip() for z in w.split(";")]))
            if not pd.isna(w)
            else w
        )

    if "Index_Keywords" in x.columns:
        logging_info("Transforming Index Keywords to lower case ...")
        x["Index_Keywords"] = x.Index_Keywords.map(
            lambda w: w.lower() if not pd.isna(w) else w
        )
        x["Index_Keywords"] = x.Index_Keywords.map(
            lambda w: ";".join(sorted([z.strip() for z in w.split(";")]))
            if not pd.isna(w)
            else w
        )

    if "Abstract" in x.columns:

        x.Abstract = x.Abstract.map(
            lambda w: w[0 : w.find("\u00a9")] if not pd.isna(w) else w
        )

    if "Times_Cited" in x.columns:
        logging_info("Removing <NA> from Times_Cited field ...")
        x["Times_Cited"] = x["Times_Cited"].map(lambda w: 0 if pd.isna(w) else w)

    if "Abb_Source_Title" in x.columns:
        logging_info("Removing '.' from Abb_Source_Title field ...")
        x["Abb_Source_Title"] = x["Abb_Source_Title"].map(
            lambda w: w.replace(".", "") if isinstance(w, str) else w
        )

    #
    # Historiograph ID and Internal cited references
    #
    if "Global_References" in x.columns:

        ##
        logging_info("Generating historiograph ID ...")
        x = x.assign(
            Historiograph_ID=x.Year.map(str)
            + "-"
            + x.groupby(["Year"], as_index=False)["Authors"].cumcount().map(str)
        )

        ##
        logging_info("Extracting local references ...")
        x["Local_References"] = [[] for _ in range(len(x))]
        for i_index, _ in enumerate(x.Title):

            title = x.Title[i_index].lower()
            year = x.Year[i_index]

            for j_index, references in enumerate(x.Global_References.tolist()):

                if pd.isna(references) is False and title in references.lower():

                    for reference in references.split(";"):

                        if title in reference.lower() and str(year) in reference:

                            x.at[j_index, "Local_References"] += [
                                x.Historiograph_ID[i_index]
                            ]
                            continue

        x["Local_References"] = x.Local_References.map(
            lambda w: pd.NA if len(w) == 0 else w
        )
        x["Local_References"] = x.Local_References.map(
            lambda w: ";".join(w), na_action="ignore"
        )

    #
    # Extract title and abstract words
    #
    if "Title" in x.columns:
        logging_info("Extracting title words ...")
        x["Title_words"] = extract_words(data=x, text=x.Title)

    if "Abstract" in x.columns:
        logging_info("Extracting abstract words ...")
        x["Abstract_words"] = extract_words(data=x, text=x.Abstract)

    #
    # Record ID
    #
    #  x['Record_ID'] = [x
    #
    #
    #      for i in range(len(x))
    #  ]

    x["ID"] = range(len(x))

    x = x.applymap(lambda w: pd.NA if isinstance(w, str) and w == "" else w)

    if output_file is None:
        return x

    x.to_csv(output_file, index=False)


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()

