import pandas as pd


from techminer.core.thesaurus import read_textfile
from techminer.core.map import map_


def apply_keywords_thesaurus(
    input_file="techminer.csv",
    thesaurus_file="keywords_thesaurus.txt",
    output_file="techminer.csv",
):

    df = pd.read_csv(input_file)

    #
    # Loads the thesaurus
    #
    th = read_textfile(thesaurus_file)
    th = th.compile_as_dict()

    #
    # Cleaning
    #
    if "Author_Keywords" in df.columns:
        df["Author_Keywords_CL"] = map_(df, "Author_Keywords", th.apply_as_dict)

    if "Index_Keywords" in df.columns:
        df["Index_Keywords_CL"] = map_(df, "Index_Keywords", th.apply_as_dict)

    if "Title_words" in df.columns:
        df["Title_words_CL"] = map_(df, "Title_words", th.apply_as_dict)

    if "Abstract_phrase_words" in df.columns:
        df["Abstract_phrase_words"] = map_(
            df, "Abstract_phrase_words", th.apply_as_dict
        )

    # if "Abstract_words" in df.columns:
    #      df["Abstract_words_CL"] = map_(df, "Abstract_words", th.apply_as_dict)

    #
    # Saves!
    #
    df.to_csv(output_file, index=False)
