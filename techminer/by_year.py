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
from IPython.display import clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout


import techminer.plots as plt
from techminer.plots import COLORMAPS

#  from ipywidgets import Box

TEXTLEN = 40


def analytics(
    data, output=0, plot=0, cmap="Greys", figsize=(6, 6), fontsize=11, **kwargs
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

        cmap ([type], optional): Colormap name. Defaults to 'Greys'.
        figsize (tuple, optional): figsize parameter for plots. Defaults to (10, 4).
        fontsize (int): Plot font size.

    Returns:
        [pandas.DataFrame or matplotlib.figure.Figure]: analytics table or plot.

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

    >>> analytics(data)[["Times_Cited", 'Num_Documents']]
          Times_Cited  Num_Documents
    Year                            
    2010           21              2
    2011           25              2
    2012           14              1
    2013            0              0
    2014            0              0
    2015            0              0
    2016           15              1

    >>> analytics(data)[['Cum_Num_Documents', 'Cum_Times_Cited', 'Avg_Times_Cited']]
          Cum_Num_Documents  Cum_Times_Cited  Avg_Times_Cited
    Year                                                     
    2010                  2               21             10.5
    2011                  4               46             12.5
    2012                  5               60             14.0
    2013                  5               60              0.0
    2014                  5               60              0.0
    2015                  5               60              0.0
    2016                  6               75             15.0

    * 1-- Num Documents by year plot.

    >>> fig = analytics(data, output=1)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/by-year-analytics-1-barplot.png')

    .. image:: images/by-year-analytics-1-barplot.png
        :width: 700px
        :align: center

    * 2-- Times Cited by year plot.

    >>> fig = analytics(data, output=2)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/by-year-analytics-2-barplot.png')

    .. image:: images/by-year-analytics-2-barplot.png
        :width: 700px
        :align: center


    * 3-- Cum Num Documents by year plot.

    >>> fig = analytics(data, output=3)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/by-year-analytics-3-barplot.png')

    .. image:: images/by-year-analytics-3-barplot.png
        :width: 700px
        :align: center


    * 4-- Cum Times Cited by year plot.

    >>> fig = analytics(data, output=4)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/by-year-analytics-4-barplot.png')

    .. image:: images/by-year-analytics-4-barplot.png
        :width: 700px
        :align: center


    * 5-- Avg Times Cited by year plot.

    >>> fig = analytics(data, output=5)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/by-year-analytics-5-barplot.png')

    .. image:: images/by-year-analytics-5-barplot.png
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
    #  result = result.reset_index()

    #
    # Output
    #
    if output == 0:
        result.pop("ID")
        return result

    if output == 1:
        values = result["Num_Documents"]
        darkness = result["Times_Cited"]
    if output == 2:
        values = result["Times_Cited"]
        darkness = result["Num_Documents"]
    if output == 3:
        values = result["Cum_Num_Documents"]
        darkness = result["Cum_Times_Cited"]
    if output == 4:
        values = result["Cum_Times_Cited"]
        darkness = result["Cum_Num_Documents"]
    if output == 5:
        values = result["Avg_Times_Cited"]
        darkness = None

    if plot == 0:
        return plt.bar(
            height=values,
            darkness=darkness,
            cmap=cmap,
            figsize=figsize,
            fontsize=fontsize,
            **kwargs,
        )
    if plot == 1:
        return plt.barh(
            width=values,
            darkness=darkness,
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


def __TAB0__(data):
    #
    #
    #  UI --- Left panel
    #
    #
    left_panel = [
        # 0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=[
                    "Analytics",
                    "Num Documents by Year",
                    "Times Cited by Year",
                    "Cum Num Documents by Year",
                    "Cum Times Cited by Year",
                    "Avg Times Cited by Year",
                ],
                layout=Layout(width="55%"),
            ),
        },
        # 1
        {
            "arg": "plot",
            "desc": "Plot:",
            "widget": widgets.Dropdown(
                options=["Bar plot", "Horizontal bar plot"], layout=Layout(width="55%"),
            ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
        },
        # 3
        {
            "arg": "sort_by",
            "desc": "Sort by:",
            "widget": widgets.Dropdown(
                options=[
                    "Year",
                    "Times_Cited",
                    "Num_Documents",
                    "Cum_Num_Documents",
                    "Cum_Times_Cited",
                    "Avg_Times_Cited",
                ],
                layout=Layout(width="55%"),
            ),
        },
        # 4
        {
            "arg": "ascending",
            "desc": "Ascending:",
            "widget": widgets.Dropdown(
                options=[True, False], layout=Layout(width="55%"),
            ),
        },
        # 5
        {
            "arg": "width",
            "desc": "Width:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), layout=Layout(width="55%"),
            ),
        },
        # 6
        {
            "arg": "height",
            "desc": "Height:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), layout=Layout(width="55%"),
            ),
        },
    ]
    #
    #
    # Logic
    #
    #
    def server(**kwargs):

        view = kwargs["view"]
        plot = kwargs["plot"]
        cmap = kwargs["cmap"]
        sort_by = kwargs["sort_by"]
        ascending = kwargs["ascending"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        left_panel[1]["widget"].disabled = True if view == "Analytics" else False
        left_panel[2]["widget"].disabled = True if view == "Analytics" else False
        left_panel[-4]["widget"].disabled = False if view == "Analytics" else True
        left_panel[-3]["widget"].disabled = False if view == "Analytics" else True
        left_panel[-2]["widget"].disabled = True if view == "Analytics" else False
        left_panel[-1]["widget"].disabled = True if view == "Analytics" else False

        view = {
            "Analytics": 0,
            "Num Documents by Year": 1,
            "Times Cited by Year": 2,
            "Cum Num Documents by Year": 3,
            "Cum Times Cited by Year": 4,
            "Avg Times Cited by Year": 5,
        }[view]

        plot = {"Bar plot": 0, "Horizontal bar plot": 1,}[plot]

        out = analytics(
            data,
            output=view,
            plot=plot,
            cmap=cmap,
            figsize=(width, height),
            fontsize=10,
        )

        if view == 0:
            if sort_by == "Year":
                out = out.sort_index(axis=0, ascending=ascending)
            else:
                out = out.sort_values(by=sort_by, ascending=ascending)

        output.clear_output()
        with output:
            return display(out)

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    #
    grid = GridspecLayout(13, 6, height="650px")  ## Marco externo al negro
    #
    # Left panel
    #
    for index in range(len(left_panel)):
        grid[index, 0] = widgets.HBox(
            [
                widgets.Label(value=left_panel[index]["desc"]),
                left_panel[index]["widget"],
            ],
            layout=Layout(
                display="flex", justify_content="flex-end", align_content="center",
            ),
        )
    #
    # Output
    #
    grid[:, 1:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )

    return grid


def app(data, tab=None):
    """Jupyter Lab dashboard.
    """
    app_title = "Analysis per Year"
    tab_titles = ["Time Analysis"]
    tab_list = [
        __TAB0__(data),
    ]

    if tab is not None:
        return AppLayout(
            header=widgets.HTML(
                value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                    app_title + " / " + tab_titles[tab]
                )
            ),
            center=tab_list[tab],
            pane_heights=["80px", "660px", 0],  # tamaño total de la ventana: Ok!
        )

    body = widgets.Tab()
    body.children = tab_list
    for i in range(len(tab_list)):
        body.set_title(i, tab_titles[i])
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                app_title
            )
        ),
        center=body,
        pane_heights=["80px", "720px", 0],
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
