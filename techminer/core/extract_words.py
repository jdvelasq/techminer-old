import pandas as pd
import datetime

import nltk
from nltk.corpus import stopwords


def extract_words(data, text):
    #
    def logging_info(msg):
        print(
            "{} - INFO - {}".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), msg
            )
        )

    ##
    ##
    ##  Keywords preparation
    ##
    ##

    #
    # 1.-- Prepare keyword list
    #
    keywords = pd.Series(data.Author_Keywords.tolist() + data.Index_Keywords.tolist())

    #
    # 2.-- Selects not NA records
    #
    keywords = keywords[keywords.map(lambda w: not pd.isna(w))]

    #
    # 3.-- create a set of unique words
    #
    keywords = keywords.tolist()
    keywords = [w for k in keywords for w in k.split(";")]
    keywords = sorted(list(set(keywords)))

    ##
    ##
    ## Keywords extraction
    ##
    ##

    #
    # 4.-- working column
    #

    text = text.map(lambda w: w.lower(), na_action="ignore")

    #
    # 5.-- replace multiple blank spaces
    #
    text = text.map(lambda w: " ".join(w.split()), na_action="ignore")

    #
    # 6.-- replace keywords in current text
    #
    for keyword in keywords:
        if " " in keyword or "_" in keyword:
            keyword_ = keyword.replace(" ", "_").replace("-", "_")
            text = text.map(lambda w: w.replace(keyword, keyword_), na_action="ignore")

    #
    # 7.-- Phrases spliting
    #
    for character in ["\n", ".", ";", ",", ":"]:
        text = text.map(lambda w: w.split(character)).explode()

    #
    # 8.-- remove punctuation
    #
    text = text.map(lambda w: w.replace(" - ", ""), na_action="ignore")

    text = text.map(
        lambda w: w.replace("-", "") if len(w) > 1 and w[-1] == "_" else w,
        na_action="ignore",
    )

    text = text.map(
        lambda w: w.translate(str.maketrans("", "", "!\"#$%&'()*+,./:;<=>?@[\\]^`{|}~"))
    )

    #
    # 9.-- split in words
    #
    text = text.map(lambda w: w.split())

    #
    # 10.-- bigram construction
    #
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

    #
    # 11.-- bigrams selection
    #
    bigrams = [
        b
        for b in bigrams
        if nltk.pos_tag(b.split(" "))[0][1] + " " + nltk.pos_tag(b.split(" "))[1][1]
        in ["NN", "JJ NN", "NN NN", "NNS", "NN NNS", "JJ NNS", "NN NNS",]
    ]

    #
    # 12.-- bigrams processing
    #
    text = text.map(lambda w: " ".join(w))
    for bigram in bigrams:
        bigram_ = bigram.replace(" ", "_").replace("-", "_")
        text = text.map(lambda w: w.replace(bigram, bigram_), na_action="ignore")
    text = text.map(lambda w: w.split())

    #
    # 13.-- remove stopwords
    #
    text = text.map(
        lambda w: [z for z in w if z not in stopwords.words("english")],
        na_action="ignore",
    )

    #
    # 10.-- word tagging and selection
    #
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

    #
    # 11.-- replace '_' by ' '
    #
    text = text.map(lambda w: [a.replace("_", " ") for a in w])

    result = pd.Series([[] for i in range(len(set(text.index.tolist())))])

    for index in set(text.index.tolist()):

        t = text[index]
        if isinstance(t, list):
            result[index] += t
        else:
            for m in t:
                result[index] += m

    print(len(result))
    result = [";".join(w) for w in result]
    return result

    #
    # 12.-- join words
    #
    result = []
    print(len(text))
    print(len(original))
    for index in set(text.index.tolist()):
        n = []
        for m in text.loc[index]:

            print(len(m), type(m))
            n.append(m)

        result.append(n)

    result = [";".join(w) for w in result]

    logging_info("Words extrated from document")
    return result

