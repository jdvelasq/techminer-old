import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core import corpus_filter, explode, record_to_HTML

#  from techminer.core.params import MULTIVALUED_COLS


def column_explorer(
    input_file="techminer.csv",
    top_n=50,
    only_abstract=False,
    clusters=None,
    cluster=None,
):

    df = pd.read_csv(input_file)

    ##
    ## Filter for cluster members
    ##
    if clusters is not None and cluster is not None:
        df = corpus_filter(data=df, clusters=clusters, cluster=cluster)

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
            "Bradford_Law_Zone",
            "Abstract_phrase_words",
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
                layout=Layout(width="98%"),
            ),
        },
        # 1
        {
            "arg": "value",
            "desc": "Term:",
            "widget": widgets.Dropdown(
                options=[],
                layout=Layout(width="98%"),
            ),
        },
        # 2
        {
            "arg": "title",
            "desc": "Title:",
            "widget": widgets.Select(
                options=[],
                layout=Layout(height="380pt", width="98%"),
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

        x = explode(df, column)

        if clusters is not None and column == clusters[0]:
            #
            # Populates value control with the terms in the cluster
            #
            left_panel[1]["widget"].options = clusters[1][cluster]

        elif top_n is not None:
            #
            # Populate value control with top_n terms
            #
            y = df.copy()
            y["Num_Documents"] = 1
            y = explode(
                y[
                    [
                        column,
                        "Num_Documents",
                        "Global_Citations",
                        "ID",
                    ]
                ],
                column,
            )
            y = y.groupby(column, as_index=True).agg(
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
            all_terms = pd.Series(x[column].unique())
            all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
            all_terms = all_terms.sort_values()
            left_panel[1]["widget"].options = all_terms
        #
        # Populate titles
        #
        keyword = left_panel[1]["widget"].value
        s = x[x[column] == keyword]
        s = s[["Global_Citations", "Title"]]
        s = s.sort_values(["Global_Citations", "Title"], ascending=[False, True])
        s = s[["Title"]].drop_duplicates()
        left_panel[2]["widget"].options = s["Title"].tolist()

        #
        # Print info from selected title
        #
        out = df[df["Title"] == left_panel[2]["widget"].value]
        out = out.reset_index(drop=True)
        out = out.iloc[0]
        output.clear_output()
        with output:
            display(
                widgets.HTML(
                    record_to_HTML(
                        out,
                        only_abstract=only_abstract,
                        keywords_to_highlight=[keyword],
                    )
                )
            )

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output().add_class("output_color")
    with output:
        display(
            widgets.interactive_output(
                server,
                args,
            )
        )

    grid = GridspecLayout(10, 4, height="706px")
    # grid[0, :] = widgets.HTML(
    #     value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
    #         "Column Explorer (Top {} terms)".format(top_n)
    #     )
    # )
    for i in range(0, len(left_panel) - 1):
        #  grid[i + 1, 0] = widgets.VBox(
        grid[i, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[i]["desc"]),
                left_panel[i]["widget"],
            ],
        )

    grid[len(left_panel) - 1 :, 0] = widgets.VBox(
        [
            widgets.Label(value=left_panel[-1]["desc"]),
            left_panel[-1]["widget"],
        ]
    )

    # grid[1:, 1:]
    grid[0:, 1:] = widgets.VBox(
        [
            output,
        ],  #  layout=Layout(height="657px", border="2px solid gray")
        layout=Layout(
            #  border="2px solid red",
            #  height="600px",
            #  width="99.5%",
            #  height="99.5%",
            margin="10px 4px 4px 4px",
            #  display="flex",
            #  align_items="flex-start",
        ),
    )

    return grid
