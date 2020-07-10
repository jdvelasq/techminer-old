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
FONTSIZE = 11

###############################################################################
##
##  APP
##
###############################################################################


def app(data, tab=None):
    return gui.APP(
        app_title="Analysis per Year",
        tab_titles=["Time Analysis"],
        tab_widgets=[TABapp0(data).run(),],
        tab=tab,
    )


class TABapp0(gui.TABapp_):
    def __init__(self, data):

        super(TABapp0, self).__init__()

        self.data_ = data

        self.panel_ = [
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
        super().create_grid()

    def gui(self, **kwargs):

        super().gui(**kwargs)

        for i in [1, 2, -2, -1]:
            self.panel_[i]["widget"].disabled = (
                True if kwargs["view"] == "Analytics" else False
            )

        for i in [
            -4,
            -3,
        ]:
            self.panel_[i]["widget"].disabled = (
                False if kwargs["view"] == "Analytics" else True
            )

    def update(self, button):
        """ 
        """
        data = self.data_[["Year", "Times_Cited", "ID"]].explode("Year")
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

        self.output_.clear_output()
        with self.output_:

            if self.view == "Analytics":
                result.pop("ID")
                if self.sort_by == "Year":
                    display(result.sort_index(axis=0, ascending=self.ascending))
                else:
                    display(
                        result.sort_values(by=self.sort_by, ascending=self.ascending)
                    )
                return

            if self.view == "Num Documents by Year":
                values = result["Num_Documents"]
                darkness = result["Times_Cited"]
            if self.view == "Times Cited by Year":
                values = result["Times_Cited"]
                darkness = result["Num_Documents"]
            if self.view == "Cum Num Documents by Year":
                values = result["Cum_Num_Documents"]
                darkness = result["Cum_Times_Cited"]
            if self.view == "Cum Times Cited by Year":
                values = result["Cum_Times_Cited"]
                darkness = result["Cum_Num_Documents"]
            if self.view == "Avg Times Cited by Year":
                values = result["Avg_Times_Cited"]
                darkness = None

            figsize = (self.width, self.height)
            if self.plot == "Bar plot":
                display(
                    plt.bar(
                        height=values,
                        darkness=darkness,
                        cmap=self.cmap,
                        figsize=figsize,
                        ylabel=self.view,
                    )
                )
            if self.plot == "Horizontal bar plot":
                display(
                    plt.barh(
                        width=values,
                        darkness=darkness,
                        cmap=self.cmap,
                        figsize=figsize,
                        xlabel=self.view,
                    )
                )


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


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
