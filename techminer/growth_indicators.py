import numpy as np
import pandas as pd

import techminer.common as cmn
import techminer.dashboard as dash
import techminer.by_term_per_year_analysis

import techminer.core.dashboard as dash
from techminer.core import DASH

from techminer.explode import explode
from techminer.bar_plot import bar_plot
from techminer.barh_plot import barh_plot
from techminer.stacked_bar import stacked_bar
from techminer.stacked_barh import stacked_barh
from techminer.core import limit_to_exclude


###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def apply(self):
        def average_growth_rate():
            #
            #         sum_{i=Y_start}^Y_end  Num_Documents[i] - Num_Documents[i-1]
            #  AGR = --------------------------------------------------------------
            #                          Y_end - Y_start + 1
            #
            #
            result = self.data.copy()
            result = explode(result[["Year", self.column, "ID"]], self.column)
            result["Num_Documents_per_Year"] = 1
            result = result.groupby([self.column, "Year"], as_index=False).agg(
                {"Num_Documents_per_Year": np.size}
            )

            years_AGR = sorted(set(result.Year))[-(self.time_window + 1) :]
            years_AGR = [years_AGR[0], years_AGR[-1]]
            result = result[result.Year.map(lambda w: w in years_AGR)]
            result = pd.pivot_table(
                result,
                columns="Year",
                index=self.column,
                values="Num_Documents_per_Year",
                fill_value=0,
            )
            result["AGR"] = 0.0
            result = result.assign(
                AGR=(result[years_AGR[1]] - result[years_AGR[0]]) / self.time_window
            )
            result.pop(years_AGR[0])
            result.pop(years_AGR[1])
            result.columns = list(result.columns)
            return result

        def average_documents_per_year():
            #
            #         sum_{i=Y_start}^Y_end  Num_Documents[i]
            #  AGR = -----------------------------------------
            #                  Y_end - Y_start + 1
            #
            result = self.data.copy()
            result = explode(result[["Year", self.column, "ID"]], self.column)
            result["Num_Documents_per_Year"] = 1
            result = result.groupby([self.column, "Year"], as_index=False).agg(
                {"Num_Documents_per_Year": np.size}
            )

            years_ADY = sorted(set(result.Year))[-self.time_window :]
            result = result[result.Year.map(lambda w: w in years_ADY)]
            result = result.groupby([self.column], as_index=False).agg(
                {"Num_Documents_per_Year": np.sum}
            )
            result = result.rename(columns={"Num_Documents_per_Year": "ADY"})
            result["ADY"] = result.ADY.map(lambda w: w / self.time_window)
            return result

        def compute_num_documents():

            result = self.data.copy()
            result = explode(result[["Year", self.column, "ID"]], self.column)
            result["Num_Documents_per_Year"] = 1
            result = result.groupby([self.column, "Year"], as_index=False).agg(
                {"Num_Documents_per_Year": np.size}
            )

            years_between = sorted(set(result.Year))[-self.time_window :]
            years_before = sorted(set(result.Year))[0 : -self.time_window]
            between = result[result.Year.map(lambda w: w in years_between)]
            before = result[result.Year.map(lambda w: w in years_before)]
            between = between.groupby([self.column], as_index=False).agg(
                {"Num_Documents_per_Year": np.sum}
            )
            between = between.rename(
                columns={
                    "Num_Documents_per_Year": "Between {}-{}".format(
                        years_between[0], years_between[-1]
                    )
                }
            )
            before = before.groupby([self.column], as_index=False).agg(
                {"Num_Documents_per_Year": np.sum}
            )
            before = before.rename(
                columns={"Num_Documents_per_Year": "Before {}".format(years_between[0])}
            )
            result = pd.merge(before, between, on=self.column)
            return result

        result = average_growth_rate()
        ady = average_documents_per_year()
        result = pd.merge(result, ady, on=self.column)
        result = result.assign(PDLY=round(result.ADY / len(self.data) * 100, 2))
        num_docs = compute_num_documents()
        result = pd.merge(result, num_docs, on=self.column)
        result = result.reset_index(drop=True)
        result = result.set_index(self.column)

        result = limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        result = cmn.add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        if self.top_by in ["Alphabetic", "Num Documents", "Times Cited"]:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.top_by, ascending=False, axis=0
            )
        elif self.top_by == "Average Growth Rate":
            result = result.sort_values(["AGR", "ADY", "PDLY"], ascending=False)
        elif self.top_by == "Average Documents per Year":
            result = result.sort_values(["ADY", "AGR", "PDLY"], ascending=False)
        elif self.top_by == "Percentage of Documents in Last Years":
            result = result.sort_values(["PDLY", "ADY", "AGR"], ascending=False)
        elif self.top_by == "Before":
            result = result.sort_values(
                [result.columns[-2], result.columns[-1]], ascending=False
            )
        elif self.top_by == "Between":
            result = result.sort_values(
                [result.columns[-1], result.columns[-2]], ascending=False
            )
        else:
            pass

        result = result.head(self.top_n)

        if self.sort_by in ["Alphabetic", "Num Documents", "Times Cited"]:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )
        else:

            if isinstance(self.sort_by, str):
                sort_by = self.sort_by.replace(" ", "_")
                sort_by = {
                    "Average_Growth_Rate": 3,
                    "Average_Documents_per_Year": 4,
                    "Percentage_of_Documents_in_Last_Years": 5,
                    "Before": 6,
                    "Between": 7,
                }[sort_by]

            if sort_by == 3:
                result = result.sort_values(
                    ["AGR", "ADY", "PDLY"], ascending=self.ascending
                )

            if sort_by == 4:
                result = result.sort_values(
                    ["ADY", "AGR", "PDLY"], ascending=self.ascending
                )

            if sort_by == 5:
                result = result.sort_values(
                    ["PDLY", "ADY", "AGR"], ascending=self.ascending
                )

            if sort_by == 6:
                result = result.sort_values(
                    [result.columns[-2], result.columns[-1]], ascending=self.ascending
                )

            if sort_by == 7:
                result = result.sort_values(
                    [result.columns[-1], result.columns[-2]], ascending=self.ascending
                )

        self.X_ = result

    def table(self):
        self.apply()
        return self.X_

    def average_growth_rate(self):
        self.apply()
        if self.plot == "bar":
            return bar_plot(
                height=self.X_.AGR, cmap=self.cmap, figsize=(self.width, self.height)
            )
        if self.plot == "barh":
            return barh_plot(
                width=self.X_.AGR, cmap=self.cmap, figsize=(self.width, self.height)
            )

    def average_documents_per_year(self):
        self.apply()
        if self.plot == "bar":
            return bar_plot(
                height=self.X_.ADY, cmap=self.cmap, figsize=(self.width, self.height)
            )
        if self.plot == "barh":
            return barh_plot(
                width=self.X_.ADY, cmap=self.cmap, figsize=(self.width, self.height)
            )

    def percentage_of_documents_in_last_years(self):
        self.apply()
        if self.plot == "bar":
            return bar_plot(
                height=self.X_.PDLY, cmap=self.cmap, figsize=(self.width, self.height)
            )
        if self.plot == "barh":
            return barh_plot(
                width=self.X_.PDLY, cmap=self.cmap, figsize=(self.width, self.height)
            )

    def num_documents(self):
        self.apply()
        if self.plot == "bar":
            return stacked_bar(
                self.X_[[self.X_.columns[-2], self.X_.columns[-1]]],
                cmap=self.cmap,
                figsize=(self.width, self.height),
            )
        if self.plot == "barh":
            return stacked_barh(
                self.X_[[self.X_.columns[-2], self.X_.columns[-1]]],
                cmap=self.cmap,
                figsize=(self.width, self.height),
            )


###############################################################################
##
##  DASHBOARD
##
###############################################################################

COLUMNS = [
    "Authors",
    "Countries",
    "Institutions",
    "Author_Keywords",
    "Index_Keywords",
    "Abstract_words_CL",
    "Abstract_words",
    "Title_words_CL",
    "Title_words",
    "Affiliations",
    "Author_Keywords_CL",
    "Index_Keywords_CL",
]


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.app_title = "Growth Indicators"
        self.menu_options = [
            "Table",
            "Average Growth Rate",
            "Average Documents per Year",
            "Percentage of Documents in Last Years",
            "Num Documents",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in self.data.columns],
            ),
            dash.dropdown(desc="Time window:", options=[2, 3, 4, 5],),
            dash.separator(text="Visualization"),
            dash.dropdown(
                desc="Top by:",
                options=[
                    "Num Documents",
                    "Times Cited",
                    "Average Growth Rate",
                    "Average Documents per Year",
                    "Percentage of Documents in Last Years",
                    "Number of Document Published",
                    "Before",
                    "Between",
                ],
            ),
            dash.top_n(),
            dash.dropdown(
                desc="Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Times Cited",
                    "Average Growth Rate",
                    "Average Documents per Year",
                    "Percentage of Documents in Last Years",
                    "Before",
                    "Between",
                ],
            ),
            dash.ascending(),
            dash.dropdown(desc="Plot:", options=["bar", "barh"],),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Table":
            self.set_disabled("Plot:")
            self.set_disabled("Colormap:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Plot:")
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

