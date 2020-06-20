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
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.plots import COLORMAPS

TEXTLEN = 40


def summary(
    data, output=0, plot=0, cmap="Greys", figsize=(10, 4), fontsize=12, **kwargs
):
    """Computes analysis by year.

    Args:
        data (pandas.DataFrame): A bibliographic dataframe.
        output (int, optional): [description]. Defaults to 0.

            * 0-- Summary dataframe.

            * 1-- Num Documents by year plot.

            * 2-- Times Cited by year plot.

            * 3-- Cum Num Documents by year plot.

            * 4-- Cum Times Cited by year plot.

            * 5-- Avg Times Cited by year plot.

        plot (int, optional): Plot type. Defaults to 0.

            * 0-- Bar plot.

            * 1-- Horizontal bar plot.

        cmap ([type], optional): Colormap name. Defaults to 'Grays'.
        figsize (tuple, optional): figsize parameter for plots. Defaults to (10, 4).
        fontsize (int): Plot font size.

    Returns:
        [pandas.DataFrame or matplotlib.figure.Figure]: Summary table or plot.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> data
       Year  Times_Cited  ID
    0  2010           10   0
    1  2010           11   1
    2  2011           12   2
    3  2011           13   3
    4  2012           14   4
    5  2016           15   5

    >>> summary(data)[['Year', "Times_Cited", 'Num_Documents', 'ID']]
       Year  Times_Cited  Num_Documents      ID
    0  2010           21              2  [0, 1]
    1  2011           25              2  [2, 3]
    2  2012           14              1     [4]
    3  2013            0              0      []
    4  2014            0              0      []
    5  2015            0              0      []
    6  2016           15              1     [5]

    >>> summary(data)[['Cum_Num_Documents', 'Cum_Times_Cited', 'Avg_Times_Cited']]
       Cum_Num_Documents  Cum_Times_Cited  Avg_Times_Cited
    0                  2               21             10.5
    1                  4               46             12.5
    2                  5               60             14.0
    3                  5               60              0.0
    4                  5               60              0.0
    5                  5               60              0.0
    6                  6               75             15.0

    * 1-- Num Documents by year plot.

    >>> fig = summary(data, output=1)
    >>> fig.savefig('sphinx/images/by-year-summary-1-barplot.png')

    .. image:: images/by-year-summary-1-barplot.png
        :width: 700px
        :align: center

    * 2-- Times Cited by year plot.

    >>> fig = summary(data, output=2)
    >>> fig.savefig('sphinx/images/by-year-summary-2-barplot.png')

    .. image:: images/by-year-summary-2-barplot.png
        :width: 700px
        :align: center


    * 3-- Cum Num Documents by year plot.

    >>> fig = summary(data, output=3)
    >>> fig.savefig('sphinx/images/by-year-summary-3-barplot.png')

    .. image:: images/by-year-summary-3-barplot.png
        :width: 700px
        :align: center


    * 4-- Cum Times Cited by year plot.

    >>> fig = summary(data, output=4)
    >>> fig.savefig('sphinx/images/by-year-summary-4-barplot.png')

    .. image:: images/by-year-summary-4-barplot.png
        :width: 700px
        :align: center


    * 5-- Avg Times Cited by year plot.

    >>> fig = summary(data, output=5)
    >>> fig.savefig('sphinx/images/by-year-summary-5-barplot.png')

    .. image:: images/by-year-summary-5-barplot.png
        :width: 700px
        :align: center

    """

    #
    # Computation
    #
    data = data[["Year", "Times_Cited", "ID"]].explode("Year")
    data["Num_Documents"] = 1
    result = data.groupby("Year", as_index=False).agg(
        {"Times_Cited": np.sum, "Num_Documents": np.size}
    )
    result = result.assign(
        ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
    )
    result["Times_Cited"] = result["Times_Cited"].map(lambda w: int(w))
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

    #
    # Output
    #
    if output == 0:
        return result

    if output == 1:
        column = "Num_Documents"
        prop_to = "Times_Cited"
    if output == 2:
        column = "Times_Cited"
        prop_to = "Num_Documents"
    if output == 3:
        column = "Cum_Num_Documents"
        prop_to = "Cum_Times_Cited"
    if output == 4:
        column = "Cum_Times_Cited"
        prop_to = "Cum_Num_Documents"
    if output == 5:
        column = "Avg_Times_Cited"
        prop_to = None

    if plot == 0:
        return plt.bar(
            data=result,
            column=column,
            prop_to=prop_to,
            cmap=cmap,
            figsize=figsize,
            fontsize=fontsize,
            **kwargs,
        )
    if plot == 1:
        return plt.barh(
            data=result,
            column=column,
            prop_to=prop_to,
            cmap=cmap,
            figsize=figsize,
            fontsize=fontsize,
            **kwargs,
        )


#
#
#  APP
#
#

WIDGET_WIDTH = "180px"
LEFT_PANEL_HEIGHT = "655px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "720px", 0]


def __APP0__(data):
    """
    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> __APP0__(data)


    """
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
                    "Times Cited by Year",
                    "Cum Num Documents by Year",
                    "Cum Times Cited by Year",
                    "Avg Times Cited by Year",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "plot",
            "desc": "Plot:",
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
            "arg": "width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 4
        {
            "arg": "height",
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
        plot = kwargs["plot"]
        cmap = kwargs["cmap"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        #
        controls[1]["widget"].disabled = True if view == "Summary" else False
        controls[2]["widget"].disabled = True if view == "Summary" else False
        controls[-1]["widget"].disabled = True if view == "Summary" else False
        controls[-2]["widget"].disabled = True if view == "Summary" else False
        #
        view = {
            "Summary": 0,
            "Num Documents by Year": 1,
            "Times Cited by Year": 2,
            "Cum Num Documents by Year": 3,
            "Cum Times Cited by Year": 4,
            "Avg Times Cited by Year": 5,
        }[view]

        plot = {"Bar plot": 0, "Horizontal bar plot": 1,}[plot]

        output.clear_output()
        with output:
            return display(
                summary(
                    data,
                    output=view,
                    plot=plot,
                    cmap=cmap,
                    figsize=(width, height),
                    fontsize=10,
                )
            )
            #

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
