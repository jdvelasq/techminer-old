import pandas as pd
import os.path

from techminer.core.thesaurus import load_file_as_dict, Thesaurus

#
# The algorithm searches in order until
# detect a match
#
NAMES = [
    "univ",
    "institut",
    "college",
    "bank",
    "banco",
    "centre",
    "center",
    "centro",
    "agency",
    "council",
    "commission",
    "politec",
    "polytechnic",
    "inc.",
    "ltd.",
    "office",
    "school",
    "department",
    "direction",
    "laboratory",
    "laboratoire",
    "colegio",
    "scuola",
    "ecole",
    "hospital",
    "association",
    "asociacion",
    "escuela",
    "company",
    "organization",
    "academy",
]


def create_institutions_thesaurus(
    input_file="techminer.csv", thesaurus_file="institutions_thesaurus.txt"
):
    #
    def search_name(w):

        ##
        ## Preprocessing
        ##
        w = w.lower().strip()

        ##
        ## Affiliation has a unique string without ','
        ##
        if len(w.split(",")) == 1:
            return w.strip().lower()

        ##
        ## Search for a possible match
        ##
        for name in NAMES:
            for elem in w.split(","):
                if name in elem:
                    selected_name = elem.strip().title()
                    selected_name = selected_name.replace(" De ", " de ")
                    selected_name = selected_name.replace(" Of ", " of ")
                    selected_name = selected_name.replace(" For ", " for ")

                    return selected_name

        ##
        ## No match found -> first part of the string
        ##
        if len(w.split(",")) == 2:
            selected_name = w.split(",")[0].strip()
            selected_name = selected_name.strip().title()
            selected_name = selected_name.replace(" De ", " de ")
            selected_name = selected_name.replace(" Of ", " of ")
            selected_name = selected_name.replace(" For ", " for ")
            return selected_name

        ##
        return w

    ##
    data = pd.read_csv(input_file)

    ##
    ## Creates a list of unique affiliations
    ##
    x = data.Affiliations
    x = x.dropna()
    x = x.map(lambda w: w.split(";"))
    x = x.explode()
    x = x.map(lambda w: w.strip())
    x = x.unique().tolist()

    ##
    ## Loads existent thesaurus
    ##
    if os.path.isfile(thesaurus_file):

        dict_ = load_file_as_dict(thesaurus_file)
        clustered_text = [word for key in dict_.keys() for word in dict_[key]]
        x = [word for word in x if word not in clustered_text]

    else:

        dict_ = {}

    ##
    ## Creates a dataframe for words and keys
    ##
    x = pd.DataFrame({"word": x, "key": x})

    ##
    ## Searches a possible name for the institution
    ##
    x["key"] = x.key.map(search_name)

    ##
    ## groups by key
    ##
    grp = x.groupby(by="key").agg({"word": list})
    result = {
        key: value for key, value in zip(grp.index.tolist(), grp["word"].tolist())
    }

    if os.path.isfile(thesaurus_file):
        result = {**result, **dict_}

    Thesaurus(result, ignore_case=False, full_match=True, use_re=False).to_textfile(
        thesaurus_file
    )
    #

