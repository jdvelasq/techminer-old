"""
Analysis by Term per Year
==================================================================================================


"""

import numpy as np
import pandas as pd

import techminer.by_year as by_year
import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.dashboard import DASH
from techminer.explode import __explode as _explode
from techminer.params import EXCLUDE_COLS

TEXTLEN = 40


class BaseModel:
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def build_table(self):

        x = self.data.copy()

        #
        # 1.-- Number of documents and times cited by term per year
        #
        x = _explode(x[["Year", self.column, "Times_Cited", "ID"]], self.column)
        x["Num_Documents"] = 1
        result = x.groupby([self.column, "Year"], as_index=False).agg(
            {"Times_Cited": np.sum, "Num_Documents": np.size}
        )
        result = result.assign(
            ID=x.groupby([self.column, "Year"]).agg({"ID": list}).reset_index()["ID"]
        )
        result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

        #
        # 2.-- Summary per year
        #
        summ = _explode(x[["Year", "Times_Cited", "ID"]], "Year")
        summ.loc[:, "Num_Documents"] = 1
        summ = summ.groupby("Year", as_index=True).agg(
            {"Times_Cited": np.sum, "Num_Documents": np.size}
        )

        #
        # 3.-- dictionaries using the year as a key
        #
        num_documents_by_year = {
            key: value for key, value in zip(summ.index, summ.Num_Documents)
        }
        times_cited_by_year = {
            key: value for key, value in zip(summ.index, summ.Times_Cited)
        }

        #
        # 4.-- indicators from ScientoPy
        #
        result["summary_documents_by_year"] = result.Year.apply(
            lambda w: num_documents_by_year[w]
        )
        result["summary_documents_by_year"] = result.summary_documents_by_year.map(
            lambda w: 1 if w == 0 else w
        )
        result["summary_times_cited_by_year"] = result.Year.apply(
            lambda w: times_cited_by_year[w]
        )
        result["summary_times_cited_by_year"] = result.summary_times_cited_by_year.map(
            lambda w: 1 if w == 0 else w
        )

        result["Perc_Num_Documents"] = 0.0
        result = result.assign(
            Perc_Num_Documents=round(
                result.Num_Documents / result.summary_documents_by_year * 100, 2
            )
        )

        result["Perc_Times_Cited"] = 0.0
        result = result.assign(
            Perc_Times_Cited=round(
                result.Times_Cited / result.summary_times_cited_by_year * 100, 2
            )
        )

        result.pop("summary_documents_by_year")
        result.pop("summary_times_cited_by_year")

        result = result.rename(
            columns={
                "Num_Documents": "Num_Documents_per_Year",
                "Times_Cited": "Times_Cited_per_Year",
                "Perc_Num_Documents": "%_Num_Documents_per_Year",
                "Perc_Times_Cited": "%_Times_Cited_per_Year",
            }
        )

        ## Limit to
        limit_to = self.limit_to
        if isinstance(limit_to, dict):
            if self.column in limit_to.keys():
                limit_to = limit_to[self.column]
            else:
                limit_to = None

        if limit_to is not None:
            result = result[result[self.column].map(lambda w: w in limit_to)]

        ## Exclude
        exclude = self.exclude
        if isinstance(exclude, dict):
            if column in exclude.keys():
                exclude = exclude[self.column]
            else:
                exclude = None

        if exclude is not None:
            result = result[result[self.column].map(lambda w: w not in exclude)]

        return result

    def table(self):
        ###
        self.apply()
        ###
        if self.cmap is not None:
            return self.X_.style.background_gradient(cmap=self.cmap, axis=0)
        return self.X_


##
##
##
##  M A T R I X
##
##
##

###############################################################################
##
##  MODEL
##
###############################################################################


class MatrixModel(BaseModel):
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        BaseModel.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )

        self.top_by = None
        self.top_n = None
        self.sort_by = None
        self.ascending = None
        self.column = None
        self.cmap = None

    def apply(self):

        result = self.build_table()

        if isinstance(self.top_by, str):
            top_by = self.top_by.replace(" ", "_")
            top_by = {
                "Num_Documents_per_Year": 0,
                "Times_Cited_per_Year": 1,
                "%_Num_Documents_per_Year": 2,
                "%_Times_Cited_per_Year": 3,
                "Num_Documents": 4,
                "Times_Cited": 5,
            }[top_by]
        else:
            top_by = self.top_by

        selected_col = {
            0: "Num_Documents_per_Year",
            1: "Times_Cited_per_Year",
            2: "%_Num_Documents_per_Year",
            3: "%_Times_Cited_per_Year",
            4: "Num_Documents_per_Year",
            5: "Times_Cited_per_Year",
        }[top_by]

        for col in [
            "Num_Documents_per_Year",
            "Times_Cited_per_Year",
            "%_Num_Documents_per_Year",
            "%_Times_Cited_per_Year",
        ]:

            if col != selected_col:
                result.pop(col)

        #
        # Table pivot
        #
        result = pd.pivot_table(
            result,
            values=selected_col,
            index="Year",
            columns=self.column,
            fill_value=0,
        )

        result = cmn.add_counters_to_axis(
            X=result, axis=1, data=self.data, column=self.column
        )

        if top_by == 4:
            ## top_by num documents
            result = cmn.sort_axis(
                data=result, num_documents=True, axis=1, ascending=False
            )
            selected_columns = result.columns[: self.top_n]
            result = result[selected_columns]

        elif top_by == 5:
            ## top_by times cited
            result = cmn.sort_axis(
                data=result, num_documents=False, axis=1, ascending=False
            )
            selected_columns = result.columns[: self.top_n]
            result = result[selected_columns]

        else:
            max = result.max(axis=0)
            max = max.sort_values(ascending=False)
            if self.top_n is not None:
                max = max.head(self.top_n)
            result = result[max.index]

        sum_years = result.sum(axis=1)
        for year, index in zip(sum_years, sum_years.index):
            if year == 0:
                result = result.drop(axis=0, labels=index)
            else:
                break

        #
        # sort_by
        #
        if self.sort_by == "Values":
            columns = result.max(axis=0)
            columns = columns.sort_values(ascending=self.ascending)
            columns = columns.index.tolist()
            result = result[columns]
        else:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=1
            )

        self.X_ = result

    def matrix(self):
        ##
        self.apply()
        ##
        if self.cmap is None:
            return self.X_
        else:
            return self.X_.style.background_gradient(cmap=self.cmap, axis=None)

    def heatmap(self):
        ##
        self.apply()
        ##
        return plt.heatmap(
            X=self.X_.transpose(), cmap=self.cmap, figsize=(self.width, self.height)
        )

    def bubble_plot(self):
        ##
        self.apply()
        ##
        return plt.bubble(
            X=self.X_.transpose(),
            darkness=None,
            cmap=self.cmap,
            figsize=(self.width, self.height),
        )

    def gant(self):
        ##
        self.apply()
        ##
        return plt.gant(X=self.X_, cmap=self.cmap, figsize=(self.width, self.height))

    def gant0(self):
        ##
        self.apply()
        ##
        return plt.gant0(x=self.X_, figsize=(self.width, self.height))


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class MatrixDASHapp(DASH, MatrixModel):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        MatrixModel.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.app_title = "Term by Year Analysis"
        self.menu_options = ["Matrix", "Heatmap", "Bubble plot", "Gant", "Gant0"]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(
                desc="Top by:",
                options=[
                    "Num Documents per Year",
                    "Times Cited per Year",
                    "% Num Documents per Year",
                    "% Times Cited per Year",
                    "Num Documents",
                    "Times Cited",
                ],
            ),
            dash.top_n(),
            dash.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Values", "Num Documents", "Times Cited"],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Matrix":
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Width:")
            self.set_enabled("Height:")


##
##
##
##  M A T R I X   L I S T
##
##
##

###############################################################################
##
##  MODEL
##
###############################################################################


class MatrixListModel(BaseModel):
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        BaseModel.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )

    def apply(self):
        #
        result = self.build_table()

        ## top_n
        if isinstance(self.top_by, str):
            top_by = self.top_by.replace(" ", "_")
            top_by = {
                "Num_Documents_per_Year": 0,
                "Times_Cited_per_Year": 1,
                "%_Num_Documents_per_Year": 2,
                "%_Times_Cited_per_Year": 3,
                "Num_Documents": 4,
                "Times_Cited": 5,
            }[top_by]
        else:
            top_by = self.top_by

        columns = {
            0: ["Num_Documents_per_Year", "Times_Cited_per_Year"],
            1: ["Times_Cited_per_Year", "Num_Documents_per_Year"],
            2: ["%_Num_Documents_per_Year", "%_Times_Cited_per_Year"],
            3: ["%_Times_Cited_per_Year", "%_Num_Documents_per_Year"],
            4: ["Num_Documents", "Times_Cited"],
            5: ["Times_Cited", "Num_Documents"],
        }[top_by]

        result.sort_values(
            columns, ascending=False, inplace=True,
        )

        if self.top_n is not None:
            result = result.head(self.top_n)
            result = result.reset_index(drop=True)

        ## sort_by
        if isinstance(self.sort_by, str):
            sort_by = self.sort_by.replace(" ", "_")
            sort_by = {
                "Alphabetic": 0,
                "Year": 1,
                "Num_Documents_per_Year": 2,
                "Times_Cited_per_Year": 3,
                "%_Num_Documents_per_Year": 4,
                "%_Times_Cited_per_Year": 5,
            }[sort_by]
        else:
            sort_by = self.sort_by

        if isinstance(self.ascending, str):
            self.ascending = {"True": True, "False": False,}[self.ascending]

        if sort_by == 0:
            result = result.sort_values([self.column], ascending=self.ascending)
        else:
            result = result.sort_values(
                {
                    1: ["Year", "Num_Documents_per_Year", "Times_Cited_per_Year"],
                    2: ["Num_Documents_per_Year", "Times_Cited_per_Year", "Year"],
                    3: ["Times_Cited_per_Year", "Num_Documents_per_Year", "Year"],
                    4: ["%_Num_Documents_per_Year", "%_Times_Cited_per_Year", "Year"],
                    5: ["%_Times_Cited_per_Year", "%_Num_Documents_per_Year", "Year"],
                }[sort_by],
                ascending=self.ascending,
            )

        ###
        result.index = result[self.column]
        result = cmn.add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )
        result[self.column] = result.index
        result = result.reset_index(drop=True)
        ###
        result.pop("ID")
        result = result[
            [
                self.column,
                "Year",
                "Num_Documents_per_Year",
                "Times_Cited_per_Year",
                "%_Num_Documents_per_Year",
                "%_Times_Cited_per_Year",
            ]
        ]
        ###
        self.X_ = result


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class MatrixListDASHapp(DASH, MatrixListModel):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        MatrixListModel.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.app_title = "Terms by Year Analysis"
        self.menu_options = [
            "Table",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(
                desc="Top by:",
                options=[
                    "Num Documents per Year",
                    "Times Cited per Year",
                    "% Num Documents per Year",
                    "% Times Cited per Year",
                ],
            ),
            dash.top_n(),
            dash.dropdown(
                desc="Sort by:",
                options=[
                    "Alphabetic",
                    "Year",
                    "Num Documents per Year",
                    "Times Cited per Year",
                    "% Num Documents per Year",
                    "% Times Cited per Year",
                ],
            ),
            dash.ascending(),
            dash.cmap(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, tab=None, years_range=None):

    if tab == 1:
        return MatrixListDASHapp(
            data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        ).run()

    return MatrixDASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

