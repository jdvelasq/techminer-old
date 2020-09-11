import pandas as pd

from techminer.core.map import map_
from techminer.core.thesaurus import read_textfile


def apply_keywords_thesaurus(
    input_file="techminer.csv",
    thesaurus_file="keywords_thesaurus.txt",
    output_file="techminer.csv",
):

    df = pd.read_csv(input_file)

    ##
    ## Loads the thesaurus
    ##
    th = read_textfile(thesaurus_file)
    th = th.compile_as_dict()

    ##
    ## Cleaning
    ##
    if "Author_Keywords" in df.columns:
        df["Author_Keywords_CL"] = map_(df, "Author_Keywords", th.apply_as_dict)

    if "Index_Keywords" in df.columns:
        df["Index_Keywords_CL"] = map_(df, "Index_Keywords", th.apply_as_dict)

    if "Author_Keywords_CL" in df.columns and "Index_Keywords_CL" in df.columns:
        df["Keywords_CL"] = (
            df.Author_Keywords_CL.map(lambda w: "" if pd.isna(w) else w)
            + ";"
            + df.Index_Keywords_CL.map(lambda w: "" if pd.isna(w) else w)
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: pd.NA if w[0] == ";" and len(w) == 1 else w
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: w[1:] if w[0] == ";" else w, na_action="ignore"
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: w[:-1] if w[-1] == ";" else w, na_action="ignore"
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: ";".join(sorted(set(w.split(";")))), na_action="ignore"
        )

    if "Title_words" in df.columns:
        df["Title_words_CL"] = map_(df, "Title_words", th.apply_as_dict)

    if "Abstract_words" in df.columns:
        df["Abstract_words_CL"] = map_(df, "Abstract_words", th.apply_as_dict)

    ##
    ## Saves!
    ##
    df.to_csv(output_file, index=False)
