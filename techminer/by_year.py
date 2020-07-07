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
import techminer.gui as gui
import techminer.plots as plt
from IPython.display import display
from ipywidgets import GridspecLayout, Layout
from techminer.plots import COLORMAPS

TEXTLEN = 40


def analytics(
    data,
    view=0,
    plot=0,
    cmap="Greys",
    sort_by="Year",
    ascending=True,
    figsize=(6, 6),
    fontsize=11,
    **kwargs
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
    if view == 0:
        result.pop("ID")

        if sort_by == "Year":
            result = result.sort_index(axis=0, ascending=ascending)
        else:
            result = result.sort_values(by=sort_by, ascending=ascending)
        return result

    if view == 1:
        values = result["Num_Documents"]
        darkness = result["Times_Cited"]
    if view == 2:
        values = result["Times_Cited"]
        darkness = result["Num_Documents"]
    if view == 3:
        values = result["Cum_Num_Documents"]
        darkness = result["Cum_Times_Cited"]
    if view == 4:
        values = result["Cum_Times_Cited"]
        darkness = result["Cum_Num_Documents"]
    if view == 5:
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
    # UI
    #
    left_panel = [
        gui.dropdown(
            desc="View:",
            options=[
                "Analytics",
                "Num Documents by Year",
                "Times Cited by Year",
                "Cum Num Documents by Year",
                "Cum Times Cited by Year",
                "Avg Times Cited by Year",
            ],
        ),
        gui.dropdown(desc="Plot:", options=["Bar plot", "Horizontal bar plot"],),
        gui.cmap(),
        gui.dropdown(
            desc="Sort by:",
            options=[
                "Year",
                "Times_Cited",
                "Num_Documents",
                "Cum_Num_Documents",
                "Cum_Times_Cited",
                "Avg_Times_Cited",
            ],
        ),
        gui.ascending(),
        gui.fig_width(),
        gui.fig_height(),
    ]
    #
    # Server
    #
    def server(**kwargs):

        kwargs["figsize"] = (int(kwargs["width"]), int(kwargs["height"]))
        del kwargs["height"], kwargs["width"]

        for i in [1, 2, -2, -1]:
            left_panel[i]["widget"].disabled = (
                True if kwargs["view"] == "Analytics" else False
            )

        for i in [
            -4,
            -3,
        ]:
            left_panel[i]["widget"].disabled = (
                False if kwargs["view"] == "Analytics" else True
            )

        kwargs["view"] = {
            "Analytics": 0,
            "Num Documents by Year": 1,
            "Times Cited by Year": 2,
            "Cum Num Documents by Year": 3,
            "Cum Times Cited by Year": 4,
            "Avg Times Cited by Year": 5,
        }[kwargs["view"]]

        kwargs["plot"] = {"Bar plot": 0, "Horizontal bar plot": 1,}[kwargs["plot"]]
        kwargs["data"] = data

        output.clear_output()
        with output:
            return display(analytics(**kwargs))

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


###############################################################################
##
##  APP
##
###############################################################################


def app(data, tab=None):
    return gui.APP(
        app_title="Analysis per Year",
        tab_titles=["Time Analysis"],
        tab_widgets=[__TAB0__(data),],
        tab=tab,
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
