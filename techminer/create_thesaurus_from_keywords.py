import pandas as pd

from techminer.core.thesaurus import text_clustering
import os.path
from techminer.core.thesaurus import load_file_as_dict, Thesaurus


def create_thesaurus_from_keywords(
    input_file="techminer.csv",
    thesaurus_file="thesaurus.txt",
    include_index_keywords=False,
):

    #
    # 1.-- Loads keywords
    #
    data = pd.read_csv(input_file)
    keywords_list = data.Author_Keywords.tolist()
    if include_index_keywords is True:
        keywords_list += data.Index_Keywords.tolist()

    if os.path.isfile(thesaurus_file):

        #
        # 2.-- Loads existent thesaurus
        #
        dict_ = load_file_as_dict(thesaurus_file)

        #
        # 3.-- Selects words to cluster
        #
        clustered_words = [word for key in dict_.keys() for word in dict_[key]]
        keywords_list = [word for word in keywords_list if word not in clustered_words]
        th = text_clustering(pd.Series(keywords_list))
        th = Thesaurus(
            x={**th._thesaurus, **dict_},
            ignore_case=True,
            full_match=False,
            use_re=False,
        )
        th.to_textfile(thesaurus_file)

    else:
        #
        # Creates a new thesaurus
        #
        text_clustering(pd.Series(keywords_list)).to_textfile(thesaurus_file)

    #

