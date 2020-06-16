"""
Analysis by Year
==================================================================================================


"""
import textwrap

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.plots import COLORMAPS
import techminer.plots as plt

TEXTLEN = 40


def summary_by_year(df):
    """Computes the number of document and the number of total citations per year.
    This funciton adds the missing years in the sequence.


    Args:
        df (pandas.DataFrame): bibliographic dataframe.


    Returns:
        pandas.DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Times_Cited  ID
    0  2010           10   0
    1  2010           11   1
    2  2011           12   2
    3  2011           13   3
    4  2012           14   4
    5  2016           15   5

    >>> summary_by_year(df)[['Year', "Times_Cited", 'Num_Documents', 'ID']]
       Year  Times_Cited  Num_Documents      ID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]

    >>> summary_by_year(df)[['Cum_Num_Documents', 'Cum_Times_Cited', 'Avg_Times_Cited']]
       Cum_Num_Documents  Cum_Times_Cited  Avg_Times_Cited
    0                  2               21             10.5
    1                  4               46             12.5
    2                  5               60             14.0
    3                  5               60              0.0
    4                  5               60              0.0
    5                  5               60              0.0
    6                  6               75             15.0

    """
    data = df[["Year", "Times_Cited", "ID"]].explode("Year")
    data["Num_Documents"] = 1
    result = data.groupby("Year", as_index=False).agg(
        {"Times_Cited": np.sum, "Num_Documents": np.size}
    )
    result = result.assign(
        ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
    )
    result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))
    years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
    result = result.set_index("Year")
    result = result.reindex(years, fill_value=0)
    result["ID"] = result["ID"].map(lambda x: [] if x == 0 else x)
    result.sort_values(
        "Year", ascending=True, inplace=True,
    )
    result["Cum_Num_Documents"] = result["Num_Documents"].cumsum()
    result["Cum_Times_Cited"] = result["Times_Cited"].cumsum()
    result["Avg_Times_Cited"] = result["Times_Cited"] / result["Num_Documents"]
    result["Avg_Times_Cited"] = result["Avg_Times_Cited"].map(
        lambda x: 0 if pd.isna(x) else round(x, 2)
    )
    result = result.reset_index()
    return result


#
#
#  Plots
#
#

FONTSIZE = 13


#
#  View:

#    Cum Documents by year
#    Times Times_Cited  year **
#    Cumulative times Times_Cited  year
#    Avg times Times_Cited  year
#


def documents_by_year_bar(
    x,
    width=0.8,
    bottom=None,
    align="center",
    style="default",
    cmap="Greys",
    figsize=(10, 6),
    **kwargs
):
    """Creates a bar plot from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Times_Cited  ID
    0  2010           10   0
    1  2010           11   1
    2  2011           12   2
    3  2011           13   3
    4  2012           14   4
    5  2016           15   5

    >>> summary_by_year(df)[['Year', "Times_Cited", 'Num_Documents', 'ID']]
       Year  Times_Cited  Num_Documents      ID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]


    >>> fig = documents_by_year_bar(x=df, cmap="Blues")
    >>> fig.savefig('sphinx/images/documents-by-year-barplot.png')

    .. image:: images/documents-by-year-barplot.png
        :width: 400px
        :align: center


    """
    #
    # Data
    #
    table = summary_by_year(x)

    #
    # Color as a function of times cited
    #
    matplotlib.rc("font", size=FONTSIZE)
    if cmap is not None:
        cmap = pyplot.cm.get_cmap(cmap)
        kwargs["color"] = [
            cmap(
                (
                    0.2
                    + 0.75
                    * (value - table.Times_Cited.min())
                    / (table.Times_Cited.max() - table.Times_Cited.min())
                )
            )
            for value in table.Times_Cited
        ]

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.bar(
        x=range(len(table)),
        height=table.Num_Documents,
        width=width,
        bottom=bottom,
        align=align,
        **({}),
        **kwargs,
    )

    ax.text(
        0, table.Num_Documents.max(), "Color darkness is proportional to times cited",
    )

    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(table.Year)
    ax.tick_params(axis="x", labelrotation=90)

    ax.set_xlabel("Year")
    ax.set_ylabel("Num_Documents")

    if style == "default":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(axis="y", color="gray", linestyle=":")

    return fig


def documents_by_year_barh(
    x,
    height=0.8,
    left=None,
    figsize=(8, 5),
    align="center",
    style="default",
    cmap=None,
    **kwargs
):
    """Creates a bar plot from a dataframe.

Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Times_Cited  ID
    0  2010           10   0
    1  2010           11   1
    2  2011           12   2
    3  2011           13   3
    4  2012           14   4
    5  2016           15   5

    >>> summary_by_year(df)[['Year', "Times_Cited", 'Num_Documents', 'ID']]
       Year  Times_Cited  Num_Documents      ID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]


    >>> fig = documents_by_year_barh(x=df, cmap="Blues")
    >>> fig.savefig('sphinx/images/documents-by-year-barhplot.png')

    .. image:: images/documents-by-year-barhplot.png
        :width: 400px
        :align: center

    """

    #
    # Data
    #
    table = summary_by_year(x)

    #
    # Color as a function of times cited
    #
    matplotlib.rc("font", size=FONTSIZE)
    if cmap is not None:
        cmap = pyplot.cm.get_cmap(cmap)
        kwargs["color"] = [
            cmap(
                (
                    0.2
                    + 0.75
                    * (value - table.Times_Cited.min())
                    / (table.Times_Cited.max() - table.Times_Cited.min())
                )
            )
            for value in table.Times_Cited
        ]

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.barh(
        y=range(len(table.Year)),
        width=table.Num_Documents,
        height=height,
        left=left,
        align=align,
        **kwargs,
    )

    ax.text(
        table.Num_Documents.max(),
        0,
        "Color darkness is proportional to times cited",
        horizontalalignment="right",
    )

    # ax.invert_yaxis()
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table.Year)
    ax.set_xlabel("Num_Documents")
    ax.set_ylabel("Year")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    ax.grid(axis="x", color="gray", linestyle=":")

    return fig


def times_cited_by_year_bar(
    x,
    width=0.8,
    bottom=None,
    align="center",
    style="default",
    cmap="Greys",
    figsize=(10, 6),
    **kwargs
):
    """Creates a bar plot from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Times_Cited  ID
    0  2010           10   0
    1  2010           11   1
    2  2011           12   2
    3  2011           13   3
    4  2012           14   4
    5  2016           15   5

    >>> summary_by_year(df)[['Year', "Times_Cited", 'Num_Documents', 'ID']]
       Year  Times_Cited  Num_Documents      ID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]


    >>> fig = times_cited_by_year_bar(x=df, cmap="Blues")
    >>> fig.savefig('sphinx/images/times_cited-by-year-barplot.png')

    .. image:: images/times_cited-by-year-barplot.png
        :width: 400px
        :align: center


    """
    #
    # Data
    #
    table = summary_by_year(x)

    #
    # Color as a function of Num_Documents
    #
    matplotlib.rc("font", size=FONTSIZE)
    if cmap is not None:
        cmap = pyplot.cm.get_cmap(cmap)
        kwargs["color"] = [
            cmap(
                (
                    0.2
                    + 0.75
                    * (value - table.Num_Documents.min())
                    / (table.Num_Documents.max() - table.Num_Documents.min())
                )
            )
            for value in table.Num_Documents
        ]

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.bar(
        x=range(len(table)),
        height=table.Times_Cited,
        width=width,
        bottom=bottom,
        align=align,
        **({}),
        **kwargs,
    )

    ax.text(
        0, table.Times_Cited.max(), "Color darkness is proportional to Num Documents",
    )

    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(table.Year)
    ax.tick_params(axis="x", labelrotation=90)

    ax.set_xlabel("Year")
    ax.set_ylabel("Times_Cited")

    if style == "default":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(axis="y", color="gray", linestyle=":")

    return fig


def times_cited_by_year_barh(
    x,
    height=0.8,
    left=None,
    figsize=(8, 5),
    align="center",
    style="default",
    cmap=None,
    **kwargs
):
    """Creates a bar plot from a dataframe.

Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Times_Cited  ID
    0  2010           10   0
    1  2010           11   1
    2  2011           12   2
    3  2011           13   3
    4  2012           14   4
    5  2016           15   5

    >>> summary_by_year(df)[['Year', "Times_Cited", 'Num_Documents', 'ID']]
       Year  Times_Cited  Num_Documents      ID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]


    >>> fig = documents_by_year_barh(x=df, cmap="Blues")
    >>> fig.savefig('sphinx/images/times-cited-by-year-barhplot.png')

    .. image:: images/times-cited-by-year-barhplot.png
        :width: 400px
        :align: center

    """

    #
    # Data
    #
    table = summary_by_year(x)

    #
    # Color as a function of times cited
    #
    matplotlib.rc("font", size=FONTSIZE)
    if cmap is not None:
        cmap = pyplot.cm.get_cmap(cmap)
        kwargs["color"] = [
            cmap(
                (
                    0.2
                    + 0.75
                    * (value - table.Num_Documents.min())
                    / (table.Num_Documents.max() - table.Num_Documents.min())
                )
            )
            for value in table.Num_Documents
        ]

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.barh(
        y=range(len(table.Year)),
        width=table.Times_Cited,
        height=height,
        left=left,
        align=align,
        **kwargs,
    )

    ax.text(
        table.Times_Cited.max(),
        0,
        "Color darkness is proportional to Num_Documents",
        horizontalalignment="right",
    )

    #  ax.invert_yaxis()
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table.Year)
    ax.set_xlabel("Times Cited")
    ax.set_ylabel("Year")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    ax.grid(axis="x", color="gray", linestyle=":")

    return fig


#
#
#  APP
#
#

WIDGET_WIDTH = "180px"
LEFT_PANEL_HEIGHT = "655px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "720px", 0]


def __APP0__(df):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=[
                    "Summary",
                    "Num Documents by Year",
                    "Cum Num Documents by Year",
                    "Times Cited by Year",
                    "Cum Times Cited by Year",
                    "Avg Times Cited by Year",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "plot_type",
            "desc": "Plot type:",
            "widget": widgets.Dropdown(
                options=["Bar plot", "Horizontal bar plot"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colors:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 4
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
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
        # Logic
        #
        view = kwargs["view"]
        plot_type = kwargs["plot_type"]
        cmap = kwargs["cmap"]
        figsize_width = int(kwargs["figsize_width"])
        figsize_height = int(kwargs["figsize_height"])
        #
        controls[1]["widget"].disabled = True if view == "Summary" else False
        controls[2]["widget"].disabled = True if view == "Summary" else False
        controls[-1]["widget"].disabled = True if view == "Summary" else False
        controls[-2]["widget"].disabled = True if view == "Summary" else False
        #
        x = summary_by_year(df)
        #
        output.clear_output()
        with output:
            #
            if view == "Summary":
                x.pop("ID")
                display(x)
                return
            #
            if view == "Documents by Year":
                x = x[["Year", "Num_Documents", "Times_Cited"]]
                if plot_type == "Bar plot":
                    display(
                        plt.bar_prop(
                            x, cmap=cmap, figsize=(figsize_width, figsize_height)
                        )
                    )
                if plot_type == "Horizontal bar plot":
                    display(
                        plt.barh_prop(
                            x, cmap=cmap, figsize=(figsize_width, figsize_height)
                        )
                    )
                return
            #
            if view == "Times Cited by Year":
                x = x[["Year", "Times_Cited", "Num_Documents"]]
                if plot_type == "Bar plot":
                    display(
                        plt.bar_prop(
                            x, cmap=cmap, figsize=(figsize_width, figsize_height)
                        )
                    )
                if plot_type == "Horizontal bar plot":
                    display(
                        plt.barh_prop(
                            x, cmap=cmap, figsize=(figsize_width, figsize_height)
                        )
                    )
                return
            #
            x = summary_by_year(df)
            if view == "Cum Num Documents by Year":
                x = x[["Year", "Cum_Num_Documents"]]
            if view == "Cum Times Cited by Year":
                x = x[["Year", "Cum_Times_Cited"]]
            if view == "Avg Times Cited by Year":
                x = x[["Year", "Avg_Times_Cited"]]
            if plot_type == "Bar plot":
                display(
                    plt.bar(x=x, cmap=cmap, figsize=(figsize_width, figsize_height))
                )
            if plot_type == "Horizontal bar plot":
                display(
                    plt.barh(x=x, cmap=cmap, figsize=(figsize_width, figsize_height))
                )
            return

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    widgets.interactive_output(
        server, args,
    )
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                    if control["desc"] not in ["Figsize"]
                ]
                + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([controls[-2]["widget"], controls[-1]["widget"],]),
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


def app(df):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__APP0__(df)]
    body.set_title(0, "Time Analysis")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Summary by Year"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
