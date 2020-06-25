import ipywidgets as widgets
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

import techminer.by_term as by_term
import pandas as pd
from techminer.explode import __explode


#
#
#  Core Authors
#
#
def core_authors(data):
    """
    """
    #
    # Numero de documentos escritos por author
    #
    z = by_term.analytics(
        data, "Authors", top_by=None, top_n=None, limit_to=None, exclude=None
    )

    authors_dict = {
        author: num_docs
        for author, num_docs in zip(z.index, z.Num_Documents)
        if not pd.isna(author)
    }

    z = z[["Num_Documents"]]
    z = z.groupby(["Num_Documents"]).size()
    w = [str(round(100 * a / sum(z), 2)) + " %" for a in z]
    z = pd.DataFrame(
        {"Num Authors": z.tolist(), "%": w, "Documents written per Author": z.index}
    )
    z = z.sort_values(["Documents written per Author"], ascending=False)
    z["Acum Num Authors"] = z["Num Authors"].cumsum()
    z["% Acum"] = [
        str(round(100 * a / sum(z["Num Authors"]), 2)) + " %"
        for a in z["Acum Num Authors"]
    ]

    m = __explode(data[["Authors", "ID"]], "Authors")
    m = m.dropna()
    m["Documents_written"] = m.Authors.map(lambda w: authors_dict[w])
    n = []
    for k in z["Documents written per Author"]:
        s = m.query("Documents_written >= " + str(k))
        s = s[["ID"]]
        s = s.drop_duplicates()
        n.append(len(s))

    k = []
    for index in range(len(n) - 1):
        k.append(n[index + 1] - n[index])
    k = [n[0]] + k
    z["Num Documents"] = k
    z["Acum Num Documents"] = n

    z = z[
        [
            "Num Authors",
            "%",
            "Acum Num Authors",
            "% Acum",
            "Documents written per Author",
            "Num Documents",
            "Acum Num Documents",
        ]
    ]

    z = z.reset_index(drop=True)
    return z


#
#
#  Panel 2
#
#
# def core_authors_app(data):
#     output = widgets.Output()
#     with output:
#         display(core_authors(data).head(50))
#     grid = GridspecLayout(10, 8)
#     grid[1:, 0:] = widgets.VBox(
#         [output], layout=Layout(height="657px", border="2px solid gray")
#     )
#     return grid
