import re
import pandas as pd
from os.path import dirname, join
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def bigram_finder(text):
    bigrams = nltk.Text(word_tokenize(text)).collocation_list()
    if len(bigrams) > 0:
        bigrams = [a + " " + b for a, b in bigrams]
        for bigram in bigrams:
            text = text.replace(bigram, bigram.replace(" ", "_"))
    return text


def extract_words(data, text):

    STOPWORDS = stopwords.words("english")

    ##
    ##  Keyword list preparation
    ##
    keywords = pd.Series(data.Author_Keywords.tolist() + data.Index_Keywords.tolist())
    keywords = keywords.dropna()
    keywords = keywords.tolist()
    keywords = [w for k in keywords for w in k.split(";")]
    keywords = sorted(list(set(keywords)))

    ##
    ##  Text normalization -- lower case
    ##
    text = text.map(lambda w: w.lower(), na_action="ignore")
    text = text.map(lambda w: re.sub(r"[\s+]", " ", w), na_action="ignore",)

    compound_keywords = [keyword for keyword in keywords if len(keyword.split()) > 1]
    compound_keywords_ = [
        keyword.replace(" ", "_").replace("-", "_") for keyword in compound_keywords
    ]
    text = text.replace(to_replace=compound_keywords, value=compound_keywords_)

    ##
    ## Collocations
    ##
    text = text.map(bigram_finder, na_action="ignore")

    ##
    ## Remove typical phrases
    ##
    module_path = dirname(__file__)
    filename = join(module_path, "../data/phrases.data")
    with open(filename, "r") as f:
        phrases = f.readlines()
    phrases = [w.replace("\n", "") for w in phrases]
    pattern = "|".join(phrases)
    text = text.map(lambda w: re.sub(pattern, "", w), na_action="ignore")

    ##
    ##  Replace chars
    ##
    for index in [8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223]:
        text = text.map(lambda w: w.replace(chr(index), ""), na_action="ignore")
    text = text.map(lambda w: w.replace(" - ", ""), na_action="ignore")

    ##
    ## Keywords extraction
    ##
    text = text.map(lambda w: word_tokenize(w), na_action="ignore")
    text = text.map(
        lambda w: [re.sub(r"[^a-zA-z_\-\s]", "", z) for z in w if z != ""],
        na_action="ignore",
    )
    text = text.map(lambda w: [word for word in w if word not in STOPWORDS])

    ##
    ##  Word tagging and selection
    ##
    text = text.map(lambda w: [word for word in w if word != ""], na_action="ignore")
    text = text.map(lambda w: nltk.pos_tag(w), na_action="ignore")
    text = text.map(
        lambda w: [
            z[0]
            for z in w
            if z[1] in ["DT", "JJ", "NN", "NNS", "RB", "VBD", "VBG", "VBP", "VBZ",]
            or "_" in z[0]
            or z[0] in keywords
        ]
    )

    ##
    ##  Checks:
    ##     Replace '_' by ' '
    ##
    text = text.map(lambda w: [a.replace("_", " ") for a in w])

    result = pd.Series([[] for i in range(len(set(text.index.tolist())))])

    for index in set(text.index.tolist()):

        t = text[index]
        if isinstance(t, list):
            result[index] += t
        else:
            for m in t:
                result[index] += m

    ##
    ##  Verification
    ##

    result = [";".join(sorted([a.strip() for a in w])) for w in result]
    result = [w for w in result if len(w) > 2]
    return result
