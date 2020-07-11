import numpy as np
import pandas as pd

import techminer.plots as plt
from IPython.display import display


class ByYear:
    def __init__(self, data):
        self.data_ = data

        #
        # Computos
        #
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
        result.pop("ID")

        self.table_ = result

    def table_view(self, sort_by, ascending):
        if sort_by == "Year":
            return self.table_.sort_index(axis=0, ascending=ascending)
        return self.table_.sort_values(by=sort_by, ascending=ascending)

    def plot(self, values, darkness, plot, cmap, figsize, label):

        if plot == "Bar plot":
            return plt.bar(
                height=values,
                darkness=darkness,
                cmap=cmap,
                figsize=figsize,
                ylabel=label,
            )

        if plot == "Horizontal bar plot":
            return plt.barh(
                width=values,
                darkness=darkness,
                cmap=cmap,
                figsize=figsize,
                xlabel=label,
            )

    def Num_Documents_by_Year(self, plot, cmap, figsize):
        values = self.table_["Num_Documents"]
        darkness = self.table_["Times_Cited"]
        return self.plot(
            values=values,
            darkness=darkness,
            plot=plot,
            cmap=cmap,
            figsize=figsize,
            label="Num Documents by Year",
        )

    def Times_Cited_by_Year(self, plot, cmap, figsize):
        values = self.table_["Times_Cited"]
        darkness = self.table_["Num_Documents"]
        return self.plot(
            values=values,
            darkness=darkness,
            plot=plot,
            cmap=cmap,
            figsize=figsize,
            label="Times Cited by Year",
        )

    def Cum_Num_Documents_by_Year(self, plot, cmap, figsize):
        values = self.table_["Cum_Num_Documents"]
        darkness = self.table_["Cum_Times_Cited"]
        return self.plot(
            values=values,
            darkness=darkness,
            plot=plot,
            cmap=cmap,
            figsize=figsize,
            label="Cum Num Documents by Year",
        )

    def Cum_Times_Cited_by_Year(self, plot, cmap, figsize):
        values = self.table_["Cum_Times_Cited"]
        darkness = self.table_["Cum_Num_Documents"]
        return self.plot(
            values=values,
            darkness=darkness,
            plot=plot,
            cmap=cmap,
            figsize=figsize,
            label="Cum Times Cited by Year",
        )

    def Avg_Times_Cited_by_Year(self, plot, cmap, figsize):
        values = self.table_["Avg_Times_Cited"]
        darkness = None
        return self.plot(
            values=values,
            darkness=darkness,
            plot=plot,
            cmap=cmap,
            figsize=figsize,
            label="Avg Times Cited by Year",
        )


import ipywidgets as widgets
from ipywidgets import AppLayout, GridspecLayout, Layout
from techminer.params import EXCLUDE_COLS
import techminer.gui as gui
from techminer.dashboard import DASH


class DASHapp(DASH):
    def __init__(self, data):

        super(DASH, self).__init__()

        self.data_ = data
        self.app_title_ = "By Year Analysis"
        self.menu_options_ = [
            "Table",
            "Num Documents by Year",
            "Times Cited by Year",
            "Cum Num Documents by Year",
            "Cum Times Cited by Year",
            "Avg Times Cited by Year",
        ]

        #  COLUMNS = sorted(
        #      [column for column in data.columns if column not in EXCLUDE_COLS]
        #  )

        self.main_panel_ = None

        self.aux_panel_ = [
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
            gui.dropdown(desc="Plot:", options=["Bar plot", "Horizontal bar plot"],),
            gui.cmap(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

        #
        self.obj_ = ByYear(data)

    def gui(self, **kwargs):

        super().gui(**kwargs)

        if self.menu_.value == self.menu_options_[0]:

            self.aux_panel_[0]["widget"].disabled = False
            self.aux_panel_[1]["widget"].disabled = False
            self.aux_panel_[2]["widget"].disabled = True
            self.aux_panel_[3]["widget"].disabled = True
            self.aux_panel_[4]["widget"].disabled = True
            self.aux_panel_[5]["widget"].disabled = True

        else:

            self.aux_panel_[0]["widget"].disabled = True
            self.aux_panel_[1]["widget"].disabled = True
            self.aux_panel_[2]["widget"].disabled = False
            self.aux_panel_[3]["widget"].disabled = False
            self.aux_panel_[4]["widget"].disabled = False
            self.aux_panel_[5]["widget"].disabled = False

    def calculate(self, button):

        with self.output_[self.tab_.selected_index]:
            display("calculate in tab " + str(self.tab_.selected_index))

    def update(self, button):

        self.output_.clear_output()
        with self.output_:

            if self.menu_.value == self.menu_options_[0]:
                display(
                    self.obj_.table_view(sort_by=self.sort_by, ascending=self.ascending)
                )

            if self.menu_.value == self.menu_options_[1]:
                display(
                    self.obj_.Num_Documents_by_Year(
                        plot=self.plot,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )

            if self.menu_.value == self.menu_options_[2]:
                display(
                    self.obj_.Times_Cited_by_Year(
                        plot=self.plot,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )

            if self.menu_.value == self.menu_options_[3]:
                display(
                    self.obj_.Cum_Num_Documents_by_Year(
                        plot=self.plot,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )

            if self.menu_.value == self.menu_options_[4]:
                display(
                    self.obj_.Cum_Times_Cited_by_Year(
                        plot=self.plot,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )

            if self.menu_.value == self.menu_options_[5]:
                display(
                    self.obj_.Avg_Times_Cited_by_Year(
                        plot=self.plot,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data):
    return DASHapp(data=data).run()

