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

        #
        # Create a new column 'Keywords'
        #
        df["Keywords_CL"] = pd.NA

        cond = df.Author_Keywords_CL.map(
            lambda w: not pd.isna(w)
        ) & df.Index_Keywords_CL.map(lambda w: not pd.isna(w))
        df.loc[cond, "Keywords_CL"] = (
            df.Author_Keywords_CL[cond] + ";" + df.Index_Keywords[cond]
        )

        cond = df.Keywords_CL.map(pd.isna) & df.Author_Keywords_CL.map(
            lambda w: not pd.isna(w)
        )
        df.loc[cond, "Keywords_CL"] = df.Author_Keywords_CL[cond]

        cond = df.Keywords_CL.map(pd.isna) & df.Index_Keywords_CL.map(
            lambda w: not pd.isna(w)
        )
        df.loc[cond, "Keywords_CL"] = df.Index_Keywords_CL[cond]

        df.Keywords_CL = df.Keywords_CL.map(
            lambda w: w if pd.isna(w) else ";".join(sorted(set(w.split(";"))))
        )

    df.to_csv(output_file, index=False)
