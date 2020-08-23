import numpy as np
import pandas as pd

import techminer.core.dashboard as dash
from techminer.core import DASH
from techminer.plots import bar_plot
from techminer.plots import barh_plot


###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data, years_range):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data

        self.ascending = True
        self.cmap = None
        self.height = None
        self.plot = None
        self.sort_by = None
        self.width = None

    def apply(self):
        ##
        data = self.data[["Year", "Times_Cited", "ID"]].explode("Year")
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
        self.X_ = result

    def table(self):
        self.apply()
        if self.sort_by == "Year":
            return self.X_.sort_index(axis=0, ascending=self.ascending)
        return self.X_.sort_values(by=self.sort_by, ascending=self.ascending)

    def plot_(self, values, darkness, label, figsize):
        self.apply()
        if self.plot == "Bar plot":
            return bar_plot(
                height=values,
                darkness=darkness,
                ylabel=label,
                figsize=figsize,
                cmap=self.cmap,
            )

        if self.plot == "Horizontal bar plot":
            return barh_plot(
                width=values,
                darkness=darkness,
                xlabel=label,
                figsize=figsize,
                cmap=self.cmap,
            )

    def num_documents_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Num_Documents"],
            darkness=self.X_["Times_Cited"],
            label="Num Documents by Year",
            figsize=(self.width, self.height),
        )

    def times_cited_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Times_Cited"],
            darkness=self.X_["Num_Documents"],
            label="Times Cited by Year",
            figsize=(self.width, self.height),
        )

    def cum_num_documents_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Cum_Num_Documents"],
            darkness=self.X_["Cum_Times_Cited"],
            label="Cum Num Documents by Year",
            figsize=(self.width, self.height),
        )

    def cum_times_cited_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Cum_Times_Cited"],
            darkness=self.X_["Cum_Num_Documents"],
            label="Cum Times Cited by Year",
            figsize=(self.width, self.height),
        )

    def avg_times_cited_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Avg_Times_Cited"],
            darkness=None,
            label="Avg Times Cited by Year",
            figsize=(self.width, self.height),
        )


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(self, data, years_range=None):
        #
        # Generic code
        #
        Model.__init__(self, data, years_range=years_range)
        DASH.__init__(self)

        #
        self.app_title = "By Year Analysis"
        self.menu_options = [
            "Table",
            "Num Documents by Year",
            "Times Cited by Year",
            "Cum Num Documents by Year",
            "Cum Times Cited by Year",
            "Avg Times Cited by Year",
        ]

        self.panel_widgets = [
            dash.separator(text="Visualization"),
            dash.dropdown(
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
            dash.ascending(),
            dash.dropdown(desc="Plot:", options=["Bar plot", "Horizontal bar plot"],),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        super().interactive_output(**kwargs)

        if self.menu == self.menu_options[0]:

            self.set_enabled("Sort by:")
            self.set_enabled("Ascending:")
            self.set_disabled("Plot:")
            self.set_disabled("Colormap:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        else:

            self.set_disabled("Sort by:")
            self.set_disabled("Ascending:")
            self.set_enabled("Plot:")
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def by_year_analysis(input_file="techminer.csv", years_range=None):

    data = pd.read_csv(input_file)
    return DASHapp(data=data, years_range=years_range).run()
