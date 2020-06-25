import ipywidgets as widgets
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

import techminer.by_term as by_term
import pandas as pd


def core_source_titles(data):
    """[summary]

    Args:
        data ([type]): [description]
    """
    m = by_term.analytics(
        data, "Source_title", top_by=None, top_n=None, limit_to=None, exclude=None
    )
    m = m[["Num_Documents"]]
    m = m.groupby(["Num_Documents"]).size()
    w = [str(round(100 * a / sum(m), 2)) + " %" for a in m]
    m = pd.DataFrame(
        {"Num Sources": m.tolist(), "%": w, "Documents published": m.index}
    )

    m = m.sort_values(["Documents published"], ascending=False)
    m["Acum Num Sources"] = m["Num Sources"].cumsum()
    m["% Acum"] = [
        str(round(100 * a / sum(m["Num Sources"]), 2)) + " %"
        for a in m["Acum Num Sources"]
    ]

    m["Tot Documents published"] = m["Num Sources"] * m["Documents published"]
    m["Num Documents"] = m["Tot Documents published"].cumsum()
    m["Tot Documents"] = m["Num Documents"].map(
        lambda w: str(round(w / m["Num Documents"].max() * 100, 2)) + " %"
    )

    bradford1 = int(len(data) / 3)
    bradford2 = 2 * bradford1

    m["Bradford's Group"] = m["Num Documents"].map(
        lambda w: 3 if w > bradford2 else (2 if w > bradford1 else 1)
    )

    m = m[
        [
            "Num Sources",
            "%",
            "Acum Num Sources",
            "% Acum",
            "Documents published",
            "Tot Documents published",
            "Num Documents",
            "Tot Documents",
            "Bradford's Group",
        ]
    ]

    m = m.reset_index(drop=True)
    return m


# def app(data):
#     output = widgets.Output()
#     with output:
#         display(core_source_titles(data))
#     grid = GridspecLayout(10, 8)
#     grid[0, :] = (
#         widgets.HTML(
#             value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
#                 "Core source titles"
#             )
#         ),
#     )
#     grid[1:, 0:] = widgets.VBox(
#         [output], layout=Layout(height="657px", border="2px solid gray")
#     )
#     return grid

