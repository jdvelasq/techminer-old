import pandas as pd


from techminer.core.thesaurus import read_textfile
from techminer.core.map import map_


def apply_thesaurus_to_keywords(
    input_file="techminer.csv",
    thesaurus_file="thesaurus-cleaned.txt",
    output_file="techminer.csv",
    include_index_keywords=False,
):

    df = pd.read_csv(input_file)
    th = read_textfile(thesaurus_file)

    th = th.compile_as_dict()

    df["Author_Keywords_CL"] = map_(df, "Author_Keywords", th.apply_as_dict)

    if include_index_keywords is True:
        df["Index_Keywords_CL"] = map_(df, "Index_Keywords", th.apply_as_dict)

    df.to_csv(output_file, index=False)
