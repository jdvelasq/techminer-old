import pandas as pd
import os.path
from os.path import dirname, join
import json
from techminer.core.thesaurus import load_file_as_dict, Thesaurus
import re
from techminer.core.extract_country_name import extract_country_name

#
# The algorithm searches in order until
# detect a match
#
NAMES = [
    "universidad",
    "universita",
    "universitas",
    "universitat",
    "universite",
    "universiteit",
    "university",
    "univesity",
    "unversitat",
    "unisversidade",
    "univerrsity",
    "institut",
    "instituto",
    "college",
    "colegio",
    "bank",
    "banco",
    "centre",
    "center",
    "centro",
    "agency",
    "agencia",
    "council",
    "commission",
    "comision",
    "consejo",
    "politec",
    "polytechnic",
    "politecnico",
    "department",
    "direction",
    "laboratory",
    "laboratoire",
    "laboratorio",
    "school",
    "skola",
    "scuola",
    "ecole",
    "escuela",
    "hospital",
    "association",
    "asociacion",
    "company",
    "organization",
    "academy",
    "academia",
    "tecnologico",
    "empresa",
    "inc.",
    "ltd.",
    "office",
    "oficina",
    "corporation",
    "corporacion",
    "ministerio",
    "technologies",
    "unidad",
    "tecnologico",
    "consorcio",
    "autoridad",
    "compania",
    "sociedad",
]

SPANISH = [
    "argentina",
    "colombia",
    "cuba",
    "ecuador",
    "guatemala",
    "honduras",
    "mexico",
    "panama",
    "peru",
    "spain",
    "venezuela",
]

PORTUGUES = ["brazil", "portugal"]


def create_institutions_thesaurus(
    input_file="techminer.csv", thesaurus_file="institutions_thesaurus.txt"
):
    #
    def clean_name(w):
        w = w.replace(".", "").lower().strip()
        w = (
            w.replace("'", "")
            .replace('"', "")
            .replace("“", "")
            .replace("”", "")
            .replace(".", "")
            .strip()
        )

        if "(" in w:
            w = w.replace(w[w.find("(") : w.find(")") + 1], "").strip()
            w = " ".join(w.split())

        return w

    #
    def search_name(w):

        ##
        ## Searchs a exact match
        ##
        for institution in VALID_NAMES:
            if institution in w:
                return institution

        ##
        ## Preprocessing
        ##
        w = clean_name(w)

        ##
        ## Searchs for a possible match
        ##
        for name in NAMES:
            for elem in w.split(","):

                if "-" in elem:
                    elem = elem.split("-")
                    elem = [e.strip() for e in elem]
                    elem = elem[0] if len(elem[0]) > len(elem[1]) else elem[1]

                if name in elem:

                    selected_name = elem.strip().lower()

                    return selected_name.lower()

        return pd.NA

    ##
    ## Valid names of institutions
    ##
    module_path = dirname(__file__)
    with open(join(module_path, "data/institutions.data"), "r") as f:
        VALID_NAMES = f.readlines()
    VALID_NAMES = [w.replace("\n", "").lower() for w in VALID_NAMES]
    VALID_NAMES = [w for w in VALID_NAMES if len(w) > 0]

    ##
    ## List of standardized country names
    ##
    module_path = dirname(__file__)
    with open(join(module_path, "data/worldmap.data"), "r") as f:
        countries = json.load(f)
    country_names = list(countries.keys())

    ##
    ## Adds missing countries to list of
    ## standardized countries
    ##
    for name in ["Singapore", "Malta", "United States"]:
        country_names.append(name)

    ##
    ## Country names to lower case
    ##
    country_names = {country.lower(): country for country in country_names}

    ##
    ## Loads techminer.csv
    ##
    data = pd.read_csv(input_file)

    ##
    ## Replace administrative regions and foreing
    ## names by country names in affiliations
    ##
    x = data.Affiliations
    x = x.map(lambda w: w.lower().strip(), na_action="ignore")

    ##
    ## Explodes the list of affiliations
    ##
    x = x.dropna()
    x = x.map(lambda w: w.split(";"))
    x = x.explode()
    x = x.map(lambda w: w.strip())
    x = x.unique().tolist()

    ##
    ## Loads the current thesaurus and
    ## select only new affiliations
    ##
    if os.path.isfile(thesaurus_file):

        dict_ = load_file_as_dict(thesaurus_file)
        clustered_text = [word for key in dict_.keys() for word in dict_[key]]
        x = [word for word in x if word not in clustered_text]

    else:

        dict_ = {}

    ##
    ## Processing of new affiliations
    ##
    x = pd.DataFrame({"affiliation": x})

    ##
    ## Extracts the country and drop rows
    ## without country
    ##
    x["country"] = x.affiliation.map(extract_country_name, na_action="ignore")

    ##
    ## Ignore institutions without country
    ##
    ignored_affiliations = x[x.country.isna()]["affiliation"].tolist()
    x = x.dropna()

    ##
    ## Searches a possible name for the institution
    ##
    x["key"] = x.affiliation.map(search_name)
    ignored_affiliations += x[x.key.isna()]["affiliation"].tolist()

    ##
    ## list of ignored affiliations for manual review
    ##
    with open("ignored_affiliations.txt", "w") as f:
        for aff in ignored_affiliations:
            print(aff, file=f)

    ##
    ## Search keys in foreign languages
    ##
    institutions = x.key.copy()
    institutions = institutions.dropna()
    institutions = institutions.tolist()

    for key, country in zip(x.key, x.country):

        if pd.isna(key) or pd.isna(country):
            continue

        aff = key.split()

        if country.lower() in SPANISH:

            ##
            ## Rule: XXX university ---> universidad XXX
            ##

            for foreign, spanish in [
                ("university", "universidad de "),
                ("university", "universidad de la "),
                ("university", "universidad "),
            ]:

                if aff[-1] == foreign:

                    new_name = spanish + " ".join(aff[:-1])
                    if new_name in institutions + VALID_NAMES:
                        x["key"] = x.key.map(
                            lambda w: new_name if w == key else w, na_action="ignore"
                        )

            ##
            ## Rule: national university of XXX ---> universidad nacional de XXX
            ##
            for foreign, spanish in [
                ("national university of", "universidad nacional de "),
                ("catholic university of", "universidad catolica de "),
                ("central university of", "universidad central de "),
                ("technical university of", "universidad tecnica de "),
                ("technological university of", "universidad tecnologica de "),
                ("autonomous university of", "universidad autonoma de "),
                ("polytechnic university of", "universidad politecnica de "),
                ("universitat politecnica de", "universidad politecnica de "),
                ("metropolitan university of", "universidad metropolitana de "),
                ("politechnic school of", "escuela politecnica de "),
                (
                    "pontifical catholic university of",
                    "pontificia universidad catolica de ",
                ),
            ]:

                foreign_len = len(foreign.split())
                if " ".join(aff[:foreign_len]) == foreign:

                    new_name = spanish + " ".join(aff[foreign_len:])
                    if new_name in institutions + VALID_NAMES:
                        x["key"] = x.key.map(
                            lambda w: new_name if w == key else w, na_action="ignore"
                        )

            ##
            ## Rule:
            ##
            for foreign, spanish in [
                ("universitat de", "universidad de "),
                ("university of", "universidad de "),
                ("university of", "universidad del "),
                ("university of", "universidad de la "),
            ]:

                if " ".join(aff[:2]) == foreign:

                    new_name = spanish + " ".join(aff[2:])
                    if new_name in institutions + VALID_NAMES:
                        x["key"] = x.key.map(
                            lambda w: new_name if w == key else w, na_action="ignore"
                        )

        if country.lower() in PORTUGUES:

            ##
            ## Rule
            ##
            for foreign, portugues in [
                ("state university of", "universidade estadual do "),
                ("state university of", "universidade estadual de "),
                ("state university of", "universidade estadual da "),
                ("state univesity of", "universidade estadual do "),
                ("state univesity of", "universidade estadual de "),
                ("state univesity of", "universidade estadual da "),
                ("federal university of", "universidade federal do "),
                ("federal university of", "universidade federal de "),
                ("federal university of", "universidade federal da "),
                ("universidad federal de", "universidade federal do "),
                ("universidad federal de", "universidade federal de "),
                ("universidad federal de", "universidade federal da "),
                ("universidad estatal de", "universidade federal do "),
                ("universidad estatal de", "universidade federal de "),
                ("universidad estatal de", "universidade federal da "),
                (
                    "pontifical catholic university of",
                    "pontificia universidade catolica do ",
                ),
            ]:

                foreign_len = len(foreign.split())
                if " ".join(aff[:foreign_len]) == foreign:
                    new_name = portugues + " ".join(aff[foreign_len:])
                    if new_name in institutions + VALID_NAMES:
                        x["key"] = x.key.map(
                            lambda w: new_name if w == key else w, na_action="ignore"
                        )

            ##
            ## Rule
            ##
            for foreign, portugues in [
                ("state university", "universidade estadual do "),
                ("state university", "universidade estadual de "),
                ("state university", "universidade estadual da "),
                ("federal university", "universidade federal do "),
                ("federal university", "universidade federal de "),
                ("federal university", "universidade federal da "),
            ]:

                if " ".join(aff[-2:]) == foreign:
                    new_name = portugues + " ".join(aff[:-2])
                    if new_name in institutions + VALID_NAMES:
                        x["key"] = x.key.map(
                            lambda w: new_name if w == key else w, na_action="ignore"
                        )

            for foreign, portugues in [
                ("university of", "universidade do "),
                ("university of", "universidade de "),
                ("university of", "universidade da "),
            ]:

                if " ".join(aff[:2]) == foreign:
                    new_name = portugues + " ".join(aff[-2:])
                    if new_name in institutions + VALID_NAMES:
                        x["key"] = x.key.map(
                            lambda w: new_name if w == key else w, na_action="ignore"
                        )

    ##
    ## Adds the country to the key
    ##
    x["key"] = (
        x.key + " (" + x.country.map(lambda w: w.lower(), na_action="ignore") + ")"
    )

    ##
    ## groups by key
    ##
    grp = x.groupby(by="key").agg({"affiliation": list})
    result = {
        key: value
        for key, value in zip(grp.index.tolist(), grp["affiliation"].tolist())
    }

    if os.path.isfile(thesaurus_file):
        result = {**result, **dict_}

    Thesaurus(result, ignore_case=False, full_match=True, use_re=False).to_textfile(
        thesaurus_file
    )

