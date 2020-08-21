import pandas as pd

from techminer.core.thesaurus import text_clustering


def create_thesaurus_from_title_words(
    input_file="techminer.csv", thesaurus_file="thesaurus-title-words-raw.txt"
):

    data = pd.read_csv(input_file)
    words_list = data.Title_words.tolist()
    text_clustering(pd.Series(words_list)).to_textfile(thesaurus_file)
