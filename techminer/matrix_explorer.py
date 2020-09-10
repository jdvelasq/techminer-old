"""
Data Viewer
==================================================================================================


"""

import numpy as np
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core import explode
from techminer.core.params import MULTIVALUED_COLS
from techminer.core import record_to_HTML


def matrix_explorer(input_file="techminer.csv", top_n=50, only_abstract=False):
    """Jupyter Lab dashboard to matrix data."""

    df = pd.read_csv(input_file)

    COLUMNS = sorted(
        [
            "Author_Keywords",
            "Author_Keywords_CL",
            "Authors",
            "Countries",
            "Country_1st_Author",
            "Index_Keywords",
            "Index_Keywords_CL",
            "Institution_1st_Author",
            "Institutions",
            "Keywords_CL",
            "Source_title",
            "Title",
            "Year",
            "Abstract_words",
            "Abstract_words_CL",
            "Title_words",
            "Title_words_CL",
        ]
    )
    #
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    left_panel = [
        # 0
        {
            "arg": "main_column",
            "desc": "Column:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in df.columns],
                layout=Layout(width="95%"),
            ),
        },
        # 1
        {
            "arg": "main_term",
            "desc": "Term in Column:",
            "widget": widgets.Dropdown(
                options=[],
                layout=Layout(width="95%"),
            ),
        },
        # 2
        {
            "arg": "by_column",
            "desc": "By column:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in df.columns],
                layout=Layout(width="95%"),
            ),
        },
        # 3
        {
            "arg": "by_term",
            "desc": "Term in By column:",
            "widget": widgets.Dropdown(
                options=[],
                layout=Layout(width="95%"),
            ),
        },
        # 4
        {
            "arg": "title",
            "desc": "Title:",
            "widget": widgets.Select(
                options=[],
                layout=Layout(height="270pt", width="95%"),
            ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        main_column = kwargs["main_column"]
        by_column = kwargs["by_column"]
        #
        # Populate main_column with top_n terms
        #
        xdf = df.copy()
        xdf["_key1_"] = xdf[main_column]
        xdf["_key2_"] = xdf[by_column]
        if main_column in MULTIVALUED_COLS:
            xdf = explode(xdf, "_key1_")
        #

        if top_n is not None:

            y = xdf.copy()
            y["Num_Documents"] = 1
            y = explode(
                y[
                    [
                        "_key1_",
                        "Num_Documents",
                        "Global_Citations",
                        "ID",
                    ]
                ],
                "_key1_",
            )
            y = y.groupby("_key1_", as_index=True).agg(
                {
                    "Num_Documents": np.sum,
                    "Global_Citations": np.sum,
                }
            )
            y["Global_Citations"] = y["Global_Citations"].map(lambda w: int(w))
            top_terms_freq = set(
                y.sort_values("Num_Documents", ascending=False).head(top_n).index
            )
            top_terms_cited_by = set(
                y.sort_values("Global_Citations", ascending=False).head(top_n).index
            )
            top_terms = sorted(top_terms_freq | top_terms_cited_by)
            left_panel[1]["widget"].options = top_terms
        else:
            top_terms = pd.Series(xdf["_key1_"].unique())
            top_terms = top_terms[top_terms.map(lambda w: not pd.isna(w))]
            top_terms = top_terms.sort_values()
            left_panel[1]["widget"].options = top_terms
        #
        # Subset selection
        #
        if by_column in MULTIVALUED_COLS:
            xdf = explode(xdf, "_key2_")
        xdf = xdf[xdf["_key1_"] == left_panel[1]["widget"].value]
        terms = sorted(pd.Series(xdf["_key2_"].dropna().unique()))
        left_panel[3]["widget"].options = terms
        #
        # Title
        #
        xdf = xdf[xdf["_key2_"] == left_panel[3]["widget"].value]
        if len(xdf):
            left_panel[4]["widget"].options = sorted(xdf["Title"].tolist())
        else:
            left_panel[4]["widget"].options = []
        #
        # Print info from selected title
        #
        out = df[df["Title"] == left_panel[4]["widget"].value]
        out = out.reset_index(drop=True)
        out = out.iloc[0]
        output.clear_output()
        text = ""
        with output:
            display(widgets.HTML(record_to_HTML(out, only_abstract=only_abstract)))

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(
            widgets.interactive_output(
                server,
                args,
            )
        )
    #
    #
    grid = GridspecLayout(10, 4, height="800px")
    grid[0, :] = widgets.HTML(
        value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
            "Matrix Explorer (Top {} terms)".format(top_n)
        )
    )
    for i in range(0, len(left_panel) - 1):
        grid[i + 1, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[i]["desc"]),
                left_panel[i]["widget"],
            ]
        )

    grid[len(left_panel) : len(left_panel) + 5, 0] = widgets.VBox(
        [
            widgets.Label(value=left_panel[-1]["desc"]),
            left_panel[-1]["widget"],
        ]
    )

    grid[1:, 1:] = widgets.VBox(
        [output],  #  layout=Layout(height="657px", border="2px solid gray")
        layout=Layout(border="2px solid gray"),
    )

    return grid


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
