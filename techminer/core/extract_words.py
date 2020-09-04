import re
import pandas as pd

import nltk
from nltk.corpus import stopwords

PHRASES = [
    "according to the results",
    "as a result",
    "for this purpose",
    "in this article",
    "in this paper",
    "in this research",
    "in this study",
    "in this work",
    "the aim of this article",
    "the aim of this paper",
    "the aim of this research",
    "the aim of this study",
    "the aim of this work",
    "the objective of this article is",
    "the objective of this article was",
    "the objective of this paper is",
    "the objective of this paper was",
    "the objective of this research is",
    "the objective of this research was",
    "the objective of this study is",
    "the objective of this study was",
    "the objective of this work is",
    "the objective of this work was",
    "the objective of this review is",
    "the objective of this revew was",
    "the present article",
    "the present paper",
    "the present research",
    "the present review",
    "the present study",
    "the present work",
    "the proposed method",
    "the results indicate",
    "the results obtained indicate",
    "the results obtained show",
    "the results obtained showed",
    "the results show",
    "the results showed",
    "this article aiming",
    "this article aims",
    "this article describes",
    "this article evaluates",
    "this article explored",
    "this article explores",
    "this article presents",
    "this article proposes",
    "this article starts",
    "this paper aiming",
    "this paper aims",
    "this paper describes",
    "this paper evaluates",
    "this paper evaluates",
    "this paper explored",
    "this paper explores",
    "this paper presents",
    "this paper proposes",
    "this paper starts",
    "this research aiming",
    "this research aims",
    "this research describes",
    "this research evaluates",
    "this research explored",
    "this research explores",
    "this research presents",
    "this research proposes",
    "this research starts",
    "this research starts",
    "this review aiming",
    "this review aims",
    "this review article starts",
    "this review describes",
    "this review evaluates",
    "this review explores",
    "this review paper starts",
    "this review presents",
    "this review proposes",
    "this review starts",
    "this review study starts",
    "this review work starts",
    "this study aiming",
    "this study aims",
    "this study describes",
    "this study evaluates",
    "this study explores",
    "this study presents",
    "this study proposes",
    "this study starts",
]


def extract_words(data, text):

    STOPWORDS = stopwords.words("english")

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
    keywords = keywords.dropna()

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
    ## Remove typical phrases
    ##
    for phrase in PHRASES:
        text = text.map(lambda w: w.replace(phrase, ""), na_action="ignore")

    ##
    ##  Replace multiple blank spaces
    ##
    for index in [8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223]:
        text = text.map(lambda w: w.replace(chr(index), ""), na_action="ignore")

    text = text.map(lambda w: w.replace(" - ", ""), na_action="ignore")
    text = text.map(lambda w: re.sub(r"[\s+]", " ", w), na_action="ignore",)

    ##
    ##  Replace compound keywords in current text
    ##
    compound_keywords = [keyword for keyword in keywords if len(keyword.split()) > 1]
    compount_keywords_ = [
        keyword.replace(" ", "_").replace("-", "_") for keyword in compound_keywords
    ]
    text = text.replace(to_replace=compound_keywords, value=compount_keywords_)

    ##
    ##  Phrases spliting
    ##
    for character in ["\n", ".", ";", ",", ":"]:
        text = text.map(lambda w: w.split(character)).explode()

    ##
    ##  Remove punctuation
    ##
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

            if phrase[i_word] in STOPWORDS or phrase[i_word + 1] in STOPWORDS:
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
        in ["JJ NN", "NN NN", "NNS", "NN NNS", "JJ NNS", "NN NNS",]
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
    text = text.map(lambda w: [z for z in w if z not in STOPWORDS], na_action="ignore",)

    ##
    ##  Remove isolated numbers and other symbols
    ##
    text = text.map(
        lambda w: [re.sub(r"[^a-zA-z_\-\s]", "", z) for z in w if z != ""],
        na_action="ignore",
    )

    text = text.map(lambda w: [a.strip() for a in w if len(a.strip()) > 1])

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

    result = [";".join(sorted([a.strip() for a in w])) for w in result]
    result = [w.replace(";the ", ";") for w in result]
    result = [w for w in result if len(w) > 2]
    return result
