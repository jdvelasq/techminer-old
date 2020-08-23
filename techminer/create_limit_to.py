import pandas as pd
import numpy as np

from techminer.core.explode import explode
from techminer.core.add_counters_to_axis import add_counters_to_axis
from techminer.core.sort_axis import sort_axis

COLUMNS = [
    "Abstract_words",
    "Abstract_words_CL",
    "Author_Keywords",
    "Author_Keywords_CL",
    "Authors",
    "Countries",
    "Country_1st_Author",
    "Index_Keywords",
    "Index_Keywords_CL",
    "Institution_1st_Author",
    "Institutions",
    "Source_title",
    "Title_words",
    "Title_words_CL",
]


def select_terms(data, column, min_occurrence, max_occurrence, max_items):

    data = data.copy()

    ##
    ## Element count
    ##
    data["Num_Documents"] = 1
    data = explode(data[[column, "Num_Documents", "Times_Cited", "ID",]], column,)
    result = data.groupby(column, as_index=True).agg(
        {"Num_Documents": np.sum, "Times_Cited": np.sum,}
    )
    result["Times_Cited"] = result["Times_Cited"].map(lambda w: int(w))

    ##
    ## Counts elements
    ##
    result = add_counters_to_axis(X=result, axis=0, data=data, column=column)
    result = sort_axis(data=result, num_documents=True, axis=0, ascending=False,)

    ##
    ## Select elements
    ##
    result = result[result.Num_Documents >= min_occurrence]
    result = result[result.Num_Documents <= max_occurrence]
    result = result.head(max_items)

    return [" ".join(t.split(" ")[:-1]) for t in result.index]


def create_limit_to(
    input_file="techminer.csv", min_occurrence=1, max_occurrence=1000, max_items=10000
):

    data = pd.read_csv(input_file)

    dict_ = {}

    for column in data.columns:
        dict_[column] = select_terms(
            data=data,
            column=column,
            min_occurrence=min_occurrence,
            max_occurrence=max_occurrence,
            max_items=max_items,
        )

    with open("limit_to.txt", "w") as file:
        for key in sorted(dict_.keys()):
            file.write(key + "\n")
            for item in dict_[key]:
                file.write("    " + item + "\n")
