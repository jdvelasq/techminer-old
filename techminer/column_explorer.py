import numpy as np
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core.explode import explode
from techminer.core.params import MULTIVALUED_COLS
from techminer.record_to_html import record_to_HTML


def app(df, top_n=50, only_abstract=False):
    """Jupyter Lab dashboard.
    """
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
            "Keywords",
            "Source_title",
            "Title",
            "Year",
            "Abstract_words",
            "Abstract_words_CL",
            "Title_words",
            "Title_words_CL",
        ]
    )

    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    left_panel = [
        # 0
        {
            "arg": "column",
            "desc": "Column:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in df.columns],
                layout=Layout(width="95%"),
            ),
        },
        # 1
        {
            "arg": "value",
            "desc": "Term:",
            "widget": widgets.Dropdown(options=[], layout=Layout(width="95%"),),
        },
        # 2
        {
            "arg": "title",
            "desc": "Title:",
            "widget": widgets.Select(
                options=[], layout=Layout(height="380pt", width="95%"),
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
        column = kwargs["column"]
        #
        # Populate value control with top_n terms
        #
        x = explode(df, column)
        if top_n is not None:

            y = df.copy()
            y["Num_Documents"] = 1
            y = explode(y[[column, "Num_Documents", "Times_Cited", "ID",]], column,)
            y = y.groupby(column, as_index=True).agg(
                {"Num_Documents": np.sum, "Times_Cited": np.sum,}
            )
            y["Times_Cited"] = y["Times_Cited"].map(lambda w: int(w))
            top_terms_freq = set(
                y.sort_values("Num_Documents", ascending=False).head(top_n).index
            )
            top_terms_cited_by = set(
                y.sort_values("Times_Cited", ascending=False).head(top_n).index
            )
            top_terms = sorted(top_terms_freq | top_terms_cited_by)
            left_panel[1]["widget"].options = top_terms
        else:
            all_terms = pd.Series(x[column].unique())
            all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
            all_terms = all_terms.sort_values()
            left_panel[1]["widget"].options = all_terms
        #
        # Populate titles
        #
        s = x[x[column] == left_panel[1]["widget"].value]
        left_panel[2]["widget"].options = sorted(s["Title"].tolist())
        #
        # Print info from selected title
        #
        out = df[df["Title"] == left_panel[2]["widget"].value]
        out = out.reset_index(drop=True)
        out = out.iloc[0]
        output.clear_output()
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
        display(widgets.interactive_output(server, args,))

    grid = GridspecLayout(10, 4, height="800px")
    grid[0, :] = widgets.HTML(
        value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
            "Column Explorer (Top {} terms)".format(top_n)
        )
    )
    for i in range(0, len(left_panel) - 1):
        grid[i + 1, 0] = widgets.VBox(
            [widgets.Label(value=left_panel[i]["desc"]), left_panel[i]["widget"],]
        )

    grid[len(left_panel) : len(left_panel) + 8, 0] = widgets.VBox(
        [widgets.Label(value=left_panel[-1]["desc"]), left_panel[-1]["widget"],]
    )

    grid[1:, 1:] = widgets.VBox(
        [output],  #  layout=Layout(height="657px", border="2px solid gray")
        layout=Layout(border="2px solid gray"),
    )

    return grid
