import re
import pandas as pd
import datetime

import nltk
from nltk.corpus import stopwords


def extract_words(data, text):

    ## ======================
    ##
    ## Keywords preparation
    ##
    ## ======================

    ##
    ##  Prepare keyword list
    ##
    keywords = pd.Series(data.Author_Keywords.tolist() + data.Index_Keywords.tolist())

    ##
    ##  Selects not NA records
    ##
    keywords = keywords[keywords.map(lambda w: not pd.isna(w))]

    ##
    ##  Create a set of unique words
    ##
    keywords = keywords.tolist()
    keywords = [w for k in keywords for w in k.split(";")]
    keywords = sorted(list(set(keywords)))

    ## ======================
    ##
    ## Keywords extraction
    ##
    ## ======================

    ##
    ##  Reduce text to lower case
    ##
    text = text.map(lambda w: w.lower(), na_action="ignore")

    ##
    ##  Replace multiple blank spaces
    ##
    text = text.map(lambda w: " ".join(w.split()), na_action="ignore")

    ##
    ##  Replace compound keywords in current text
    ##
    compound_keywords = [keyword for keyword in keywords if len(keyword.split()) > 1]
    compount_keywords_ = [
        keyword.replace(" ", "_").replace("-", "_") for keyword in compound_keywords
    ]
    text = text.replace(to_replace=compound_keywords, value=compount_keywords_)

    # for keyword in keywords:
    #     if " " in keyword or "_" in keyword:
    #         keyword_ = keyword.replace(" ", "_").replace("-", "_")
    #         text = text.map(lambda w: w.replace(keyword, keyword_), na_action="ignore")

    ##
    ##  Phrases spliting
    ##
    for character in ["\n", ".", ";", ",", ":"]:
        text = text.map(lambda w: w.split(character)).explode()

    ##
    ##  Remove punctuation
    ##
    text = text.map(lambda w: w.replace(" - ", ""), na_action="ignore")

    text = text.map(
        lambda w: w.replace("-", "") if len(w) > 1 and w[-1] == "_" else w,
        na_action="ignore",
    )

    text = text.map(
        lambda w: w.translate(str.maketrans("", "", "!\"#$%&'()*+,./:;<=>?@[\\]^`{|}~"))
    )

    ##
    ##  Split in words
    ##
    text = text.map(lambda w: w.split())

    ##
    ##  Bigram construction
    ##
    bigrams = []
    for phrase in text:
        for i_word, _ in enumerate(phrase):

            if i_word == len(phrase) - 1:
                continue

            if phrase[i_word] in stopwords.words("english") or phrase[
                i_word + 1
            ] in stopwords.words("english"):
                continue

            if "_" in phrase[i_word] or "_" in phrase[i_word + 1]:
                continue

            if phrase[i_word] in keywords or phrase[i_word + 1] in keywords:
                continue

            bigram = phrase[i_word] + " " + phrase[i_word + 1]
            bigrams.append(bigram)

    ##
    ##  Bigrams selection
    ##
    bigrams = [
        b
        for b in bigrams
        if nltk.pos_tag(b.split(" "))[0][1] + " " + nltk.pos_tag(b.split(" "))[1][1]
        in ["NN", "JJ NN", "NN NN", "NNS", "NN NNS", "JJ NNS", "NN NNS",]
    ]

    ##
    ##  Bigrams processing
    ##
    text = text.map(lambda w: " ".join(w))
    for bigram in bigrams:
        bigram_ = bigram.replace(" ", "_").replace("-", "_")
        text = text.map(lambda w: w.replace(bigram, bigram_), na_action="ignore")
    text = text.map(lambda w: w.split())

    ##
    ##  Remove stopwords
    ##
    text = text.map(
        lambda w: [z for z in w if z not in stopwords.words("english")],
        na_action="ignore",
    )

    ##
    ##  Remove isolated numbers and other symbols
    ##
    text = text.map(
        lambda w: [re.sub(r"[^a-zA-z_\-\s]", "", z) for z in w if z != ""],
        na_action="ignore",
    )

    #  text = text.map(
    #      lambda w: [z for z in w if not z.replace("-", "").isdigit()],
    #     na_action="ignore",
    # )

    # text = text.map(
    #     lambda w: [z[1:] if z[0] == "-" and len(z) > 1 else z for z in w],
    #     na_action="ignore",
    # )

    # text = text.map(lambda w: [z.replace("#", "") for z in w], na_action="ignore",)
    text = text.map(
        lambda w: [z.replace(chr(8220), "") for z in w], na_action="ignore",
    )
    text = text.map(
        lambda w: [z.replace(chr(8221), "") for z in w], na_action="ignore",
    )
    text = text.map(
        lambda w: [z.replace(chr(8212), "") for z in w], na_action="ignore",
    )
    # text = text.map(
    #     lambda w: [z.replace(chr(8212), "") for z in w], na_action="ignore",
    # )
    #  text = text.map(
    #      lambda w: [a[1:] if a[0] == chr(8212) and len(a) > 2 else a for a in w],
    #      na_action="ignore",
    #  )
    # text = text.map(
    #     lambda w: [a for a in w if not a.replace("-", "").isdigit()], na_action="ignore"
    # )
    # text = text.map(lambda w: [a.replace('"', "") for a in w])
    text = text.map(lambda w: [a for a in w if len(a.strip()) > 1])

    ##
    ##  Word tagging and selection
    ##
    text = text.map(lambda w: nltk.pos_tag(w), na_action="ignore")
    text = text.map(
        lambda w: [
            z[0]
            for z in w
            if z[1] in ["NN", "JJ NN", "NN NN", "NNS", "NN NNS", "JJ NNS", "NN NNS",]
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

    result = [";".join(w) for w in result]
    return result
