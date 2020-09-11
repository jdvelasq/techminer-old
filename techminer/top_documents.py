import pandas as pd

from techminer.core import corpus_filter


def top_documents(input_file="techminer.csv", top_n=10, clusters=None, cluster=None):

    data = pd.read_csv(input_file)

    #
    # Filter for cluster members
    #
    if clusters is not None and cluster is not None:
        data = corpus_filter(data=data, clusters=clusters, cluster=cluster)

    data = data.sort_values("Global_Citations", ascending=False).head(top_n)
    data = data.reset_index(drop=True)
    for i in range(len(data)):
        print(
            data.Authors[i].replace(";", ", ")
            + ". "
            + str(data.Year[i])
            + ". "
            + data.Title[i]
            + "\t"
            + str(int(data.Global_Citations[i]))
        )
