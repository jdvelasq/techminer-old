import pandas as pd


from techminer.core.thesaurus import read_textfile
from techminer.core.map import map_
from techminer.core.logging_info import logging_info


def apply_institutions_thesaurus(
    input_file="techminer.csv",
    thesaurus_file="institutions_thesaurus.txt",
    output_file="techminer.csv",
):

    data = pd.read_csv(input_file)

    ##
    ## Loads the thesaurus
    ##
    th = read_textfile(thesaurus_file)
    th = th.compile_as_dict()

    ##
    ## Copy affiliations to institutions
    ##
    data["Institutions"] = data.Affiliations

    ##
    ## Cleaning
    ##
    logging_info("Extract and cleaning institutions.")
    data["Institutions"] = map_(data, "Institutions", th.apply_as_dict)

    logging_info("Extracting institution of first author ...")
    data["Institution_1st_Author"] = data.Institutions.map(
        lambda w: w.split(";")[0] if isinstance(w, str) else w
    )

    ##
    ## Finish!
    ##
    data.to_csv(output_file, index=False)
