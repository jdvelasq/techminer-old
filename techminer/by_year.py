import numpy as np
import pandas as pd

import techminer.dashboard as dash
import techminer.plots as plt
from techminer.dashboard import DASH


class Model:
    def __init__(self, data):
        #
        self.data = data
        #
        self.ascending = True
        self.cmap = None
        self.height = None
        self.plot = None
        self.sort_by = None
        self.width = None
        #
        self.fit()

    def fit(self):
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
        if self.sort_by == "Year":
            return self.X_.sort_index(axis=0, ascending=self.ascending)
        return self.X_.sort_values(by=self.sort_by, ascending=self.ascending)

    def plot_(self, values, darkness, label, figsize):

        if self.plot == "Bar plot":
            return plt.bar(
                height=values,
                darkness=darkness,
                ylabel=label,
                figsize=figsize,
                cmap=self.cmap,
            )

        if self.plot == "Horizontal bar plot":
            return plt.barh(
                width=values,
                darkness=darkness,
                xlabel=label,
                figsize=figsize,
                cmap=self.cmap,
            )

    def num_documents_by_year(self):
        return self.plot_(
            values=self.X_["Num_Documents"],
            darkness=self.X_["Times_Cited"],
            label="Num Documents by Year",
            figsize=(self.width, self.height),
        )

    def times_cited_by_year(self):
        return self.plot_(
            values=self.X_["Times_Cited"],
            darkness=self.X_["Num_Documents"],
            label="Times Cited by Year",
            figsize=(self.width, self.height),
        )

    def cum_num_documents_by_year(self):
        return self.plot_(
            values=self.X_["Cum_Num_Documents"],
            darkness=self.X_["Cum_Times_Cited"],
            label="Cum Num Documents by Year",
            figsize=(self.width, self.height),
        )

    def cum_times_cited_by_year(self):
        return self.plot_(
            values=self.X_["Cum_Times_Cited"],
            darkness=self.X_["Cum_Num_Documents"],
            label="Cum Times Cited by Year",
            figsize=(self.width, self.height),
        )

    def avg_times_cited_by_year(self):
        return self.plot_(
            values=self.X_["Avg_Times_Cited"],
            darkness=None,
            label="Avg Times Cited by Year",
            figsize=(self.width, self.height),
        )


class DASHapp(DASH, Model):
    def __init__(self, data, year_range=None):
        """Dashboard app"""

        if year_range is not None:
            initial_year, final_year = year_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        Model.__init__(self, data)
        DASH.__init__(self)

        self.data = data
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


def app(data, year_range=None):
    return DASHapp(data=data, year_range=year_range).run()
