import pandas as pd

from techminer.core.thesaurus import text_clustering


def create_thesaurus_from_keywords(
    input_file="techminer.csv",
    thesaurus_file="thesaurus-keywords-raw.txt",
    include_index_keywords=False,
):

    data = pd.read_csv(input_file)

    keywords_list = data.Author_Keywords.tolist()
    if include_index_keywords is True:
        keywords_list += data.Index_Keywords.tolist()

    text_clustering(pd.Series(keywords_list)).to_textfile(thesaurus_file)
