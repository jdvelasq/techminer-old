"""
Analysis by Term
==========================================================================

"""
import numpy as np
import pandas as pd
import ipywidgets as widgets


import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.dashboard import DASH
from techminer.explode import __explode as _explode
from techminer.params import EXCLUDE_COLS

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude):
        #
        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude
        #
        self.column = None
        self.top_by = None
        self.top_n = None
        self.sort_by = None
        self.ascending = None
        self.cmap = None
        self.height = None
        self.width = None
        self.view = None

    def core_source_titles(self):
        """Compute source title statistics """

        x = self.data.copy()
        x["Num_Documents"] = 1
        x = _explode(x[["Source_title", "Num_Documents", "ID",]], "Source_title",)
        m = x.groupby("Source_title", as_index=True).agg({"Num_Documents": np.sum,})
        m = m[["Num_Documents"]]
        m = m.groupby(["Num_Documents"]).size()
        w = [str(round(100 * a / sum(m), 2)) + " %" for a in m]
        m = pd.DataFrame(
            {"Num Sources": m.tolist(), "%": w, "Documents published": m.index}
        )

        m = m.sort_values(["Documents published"], ascending=False)
        m["Acum Num Sources"] = m["Num Sources"].cumsum()
        m["% Acum"] = [
            str(round(100 * a / sum(m["Num Sources"]), 2)) + " %"
            for a in m["Acum Num Sources"]
        ]

        m["Tot Documents published"] = m["Num Sources"] * m["Documents published"]
        m["Num Documents"] = m["Tot Documents published"].cumsum()
        m["Tot Documents"] = m["Num Documents"].map(
            lambda w: str(round(w / m["Num Documents"].max() * 100, 2)) + " %"
        )

        bradford1 = int(len(self.data) / 3)
        bradford2 = 2 * bradford1

        m["Bradford's Group"] = m["Num Documents"].map(
            lambda w: 3 if w > bradford2 else (2 if w > bradford1 else 1)
        )

        m = m[
            [
                "Num Sources",
                "%",
                "Acum Num Sources",
                "% Acum",
                "Documents published",
                "Tot Documents published",
                "Num Documents",
                "Tot Documents",
                "Bradford's Group",
            ]
        ]

        m = m.reset_index(drop=True)
        return m

    def core_authors(self):
        """
        """

        x = self.data.copy()

        #
        # 1.-- Num_Documents per Author
        #
        x["Num_Documents"] = 1
        x = _explode(x[["Authors", "Num_Documents", "ID",]], "Authors",)
        result = x.groupby("Authors", as_index=True).agg({"Num_Documents": np.sum,})
        z = result
        authors_dict = {
            author: num_docs
            for author, num_docs in zip(z.index, z.Num_Documents)
            if not pd.isna(author)
        }

        #
        # 2.-- Num Authors x Documents written per Author
        #
        z = z[["Num_Documents"]]
        z = z.groupby(["Num_Documents"]).size()
        w = [str(round(100 * a / sum(z), 2)) + " %" for a in z]
        z = pd.DataFrame(
            {"Num Authors": z.tolist(), "%": w, "Documents written per Author": z.index}
        )
        z = z.sort_values(["Documents written per Author"], ascending=False)
        z["Acum Num Authors"] = z["Num Authors"].cumsum()
        z["% Acum"] = [
            str(round(100 * a / sum(z["Num Authors"]), 2)) + " %"
            for a in z["Acum Num Authors"]
        ]
        m = _explode(self.data[["Authors", "ID"]], "Authors")
        m = m.dropna()
        m["Documents_written"] = m.Authors.map(lambda w: authors_dict[w])

        n = []
        for k in z["Documents written per Author"]:
            s = m.query("Documents_written >= " + str(k))
            s = s[["ID"]]
            s = s.drop_duplicates()
            n.append(len(s))

        k = []
        for index in range(len(n) - 1):
            k.append(n[index + 1] - n[index])
        k = [n[0]] + k
        z["Num Documents"] = k
        z["% Num Documents"] = [str(round(i / max(n) * 100, 2)) + "%" for i in k]
        z["Acum Num Documents"] = n
        z["% Acum Num Documents"] = [str(round(i / max(n) * 100, 2)) + "%" for i in n]

        z = z[
            [
                "Num Authors",
                "%",
                "Acum Num Authors",
                "% Acum",
                "Documents written per Author",
                "Num Documents",
                "% Num Documents",
                "Acum Num Documents",
                "% Acum Num Documents",
            ]
        ]

        z = z.reset_index(drop=True)
        return z

    def top_documents(self):
        """Returns the top 50 documents by Times Cited."""
        data = self.data
        data = data.sort_values(["Times_Cited", "Year"], ascending=[False, True])
        data = data.head(50)
        data["Times_Cited"] = data.Times_Cited.map(lambda w: int(w))
        data = data.reset_index(drop=True)
        data = data.sort_values(["Times_Cited", "Title"], ascending=[False, True])
        data = data[["Authors", "Year", "Title", "Source_title", "Times_Cited"]]
        data["Times_Cited"] = data.Times_Cited.map(lambda w: int(w))
        data = data.reset_index(drop=True)
        return data

    def worldmap(self):
        x = self.data.copy()
        x["Num_Documents"] = 1
        x = _explode(
            x[[self.column, "Num_Documents", "Times_Cited", "ID",]], self.column,
        )
        result = x.groupby(self.column, as_index=True).agg(
            {"Num_Documents": np.sum, "Times_Cited": np.sum,}
        )
        top_by = self.top_by.replace(" ", "_")
        return plt.worldmap(
            x=result[top_by], figsize=(self.width, self.height), cmap=self.cmap,
        )

    def single_multiple_publication(self):
        x = self.data.copy()
        x["SD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
        )
        x["MD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
        )
        x = _explode(x[[self.column, "SD", "MD", "ID",]], self.column,)
        result = x.groupby(self.column, as_index=False).agg(
            {"SD": np.sum, "MD": np.sum,}
        )
        result["SMR"] = [
            round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)
        ]
        result = result.set_index(self.column)

        ## limit to / exclude options
        result = cmn.limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ## counters in axis names
        result = cmn.add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ## Top by / Top N
        result = cmn.sort_by_axis(
            data=result, sort_by=self.top_by, ascending=False, axis=0
        )
        result = result.head(self.top_n)

        ## Sort by
        if self.sort_by in result.columns:
            result = result.sort_values(self.sort_by, ascending=self.ascending)
        else:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )

        if self.view == "Table":
            return result

        if self.view == "Bar plot":
            return plt.stacked_bar(
                X=result[["SD", "MD"]],
                cmap=self.cmap,
                ylabel="Num Documents",
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            return plt.stacked_barh(
                X=result[["SD", "MD"]],
                cmap=self.cmap,
                xlabel="Num Documents",
                figsize=(self.width, self.height),
            )

    def impact(self):
        x = self.data.copy()
        last_year = x.Year.max()
        x["Num_Documents"] = 1
        x["First_Year"] = x.Year
        x = _explode(
            x[[self.column, "Num_Documents", "Times_Cited", "First_Year", "ID"]],
            self.column,
        )
        result = x.groupby(self.column, as_index=False).agg(
            {"Num_Documents": np.sum, "Times_Cited": np.sum, "First_Year": np.min,}
        )
        result["Last_Year"] = last_year
        result = result.assign(Years=result.Last_Year - result.First_Year + 1)
        result = result.assign(Times_Cited_per_Year=result.Times_Cited / result.Years)
        result["Times_Cited_per_Year"] = result["Times_Cited_per_Year"].map(
            lambda w: round(w, 2)
        )
        result = result.assign(
            Avg_Times_Cited=result.Times_Cited / result.Num_Documents
        )
        result["Avg_Times_Cited"] = result["Avg_Times_Cited"].map(lambda w: round(w, 2))

        result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

        #
        # Indice H
        #
        z = x[[self.column, "Times_Cited", "ID"]].copy()
        z = (
            x.assign(
                rn=x.sort_values("Times_Cited", ascending=False)
                .groupby(self.column)
                .cumcount()
                + 1
            )
        ).sort_values(
            [self.column, "Times_Cited", "rn"], ascending=[False, False, True]
        )
        z["rn2"] = z.rn.map(lambda w: w * w)

        q = z.query("Times_Cited >= rn")
        q = q.groupby(self.column, as_index=False).agg({"rn": np.max})
        h_dict = {key: value for key, value in zip(q[self.column], q.rn)}

        result["H_index"] = result[self.column].map(
            lambda w: h_dict[w] if w in h_dict.keys() else 0
        )

        #
        # indice M
        #
        result = result.assign(M_index=result.H_index / result.Years)
        result["M_index"] = result["M_index"].map(lambda w: round(w, 2))

        #
        # indice G
        #
        q = z.query("Times_Cited >= rn2")
        q = q.groupby(self.column, as_index=False).agg({"rn": np.max})
        h_dict = {key: value for key, value in zip(q[self.column], q.rn)}
        result["G_index"] = result[self.column].map(
            lambda w: h_dict[w] if w in h_dict.keys() else 0
        )

        ## limit to / exclude options
        result = cmn.limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ## Top by / Top N
        top = self.top_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        if top in result.columns:
            result = result.sort_values(top, ascending=False)
        else:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.top_by, ascending=False, axis=0
            )
        result = result.head(self.top_n)

        ## counters in axis names
        result = result.set_index(self.column)
        result = cmn.add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ## Sort by
        sort_by = self.sort_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        if sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=self.ascending)
        else:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )

        if self.view == "Table":
            result.pop("Num_Documents")
            result.pop("Times_Cited")
            result.pop("First_Year")
            result.pop("Last_Year")
            result.pop("Years")
            return result

        if self.view == "Bar plot":
            top_by = self.top_by.replace(" ", "_")
            return plt.bar(
                height=result[top_by],
                cmap=self.cmap,
                ylabel=self.top_by,
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            top_by = self.top_by.replace(" ", "_")
            return plt.barh(
                width=result[top_by],
                cmap=self.cmap,
                xlabel=self.top_by,
                figsize=(self.width, self.height),
            )

    def general(self):
        x = self.data.copy()
        x["Num_Documents"] = 1
        x = _explode(
            x[[self.column, "Num_Documents", "Times_Cited", "ID",]], self.column,
        )
        result = x.groupby(self.column, as_index=True).agg(
            {"Num_Documents": np.sum, "Times_Cited": np.sum,}
        )
        result = cmn.limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        result["Times_Cited"] = result["Times_Cited"].map(lambda w: int(w))

        result = result.reset_index()
        result = result.reset_index(drop=True)
        top_by = self.top_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        result = result.sort_values(top_by, ascending=False)
        result = result.head(self.top_n)

        if self.sort_by == "Alphabetic":
            result = result.sort_values(self.column, ascending=self.ascending)
        else:
            sort_by = self.sort_by.replace(" ", "_").replace("-", "_")
            result = result.sort_values(sort_by, ascending=self.ascending)
        result.index = list(range(len(result)))

        if self.view == "Table":
            return result

        result = result.set_index(self.column)
        if self.top_by == "Num Documents":
            values = result.Num_Documents
            darkness = result.Times_Cited
        else:
            values = result.Times_Cited
            darkness = result.Num_Documents

        if self.view == "Bar plot":
            return plt.bar(
                height=values,
                darkness=darkness,
                cmap=self.cmap,
                ylabel=self.top_by,
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            return plt.barh(
                width=values,
                darkness=darkness,
                cmap=self.cmap,
                xlabel=self.top_by,
                figsize=(self.width, self.height),
            )
        if self.view == "Pie plot":
            return plt.pie(
                x=values,
                darkness=darkness,
                cmap=self.cmap,
                figsize=(self.width, self.height),
            )

        if self.view == "Wordcloud":
            return plt.wordcloud(
                x=values,
                darkness=darkness,
                cmap=self.cmap,
                figsize=(self.width, self.height),
            )

        if self.view == "Treemap":
            return plt.treemap(
                x=values,
                darkness=darkness,
                cmap=self.cmap,
                figsize=(self.width, self.height),
            )

    def list_of_core_source_titles(self):

        x = self.data.copy()
        x["Num_Documents"] = 1
        x = _explode(x[["Source_title", "Num_Documents", "ID",]], "Source_title",)
        m = x.groupby("Source_title", as_index=True).agg({"Num_Documents": np.sum,})
        m = m[["Num_Documents"]]
        m = m.sort_values(by="Num_Documents", ascending=False)
        m["Cum_Num_Documents"] = m.Num_Documents.cumsum()
        m = m[m.Cum_Num_Documents <= int(len(self.data) / 3)]
        HTML = "1 st. Bradford' Group<br>"
        for value in m.Num_Documents.unique():
            n = m[m.Num_Documents == value]
            HTML += "======================================================<br>"
            HTML += "Num Documents Published:" + str(value) + "<br>"
            HTML += "<br>"
            for source in n.index:
                HTML += "    " + source + "<br>"
            HTML += "<br><br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None):
        """Dashboard app"""

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.data = data
        self.app_title = "Terms Analysis"
        self.menu_options = [
            "General",
            "Worldmap",
            "Impact",
            "Single/Multiple publication",
            "Core authors",
            "Core source titles",
            "Top documents",
            "List of core source titles",
        ]
        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(
                desc="View:",
                options=[
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                    "Pie plot",
                    "Wordcloud",
                    "Worldmap",
                    "Treemap",
                    "S/D Ratio (bar)",
                    "S/D Ratio (barh)",
                ],
            ),
            dash.dropdown(
                desc="Top by:",
                options=[
                    "Num Documents",
                    "Times Cited",
                    "Frac Num Documents",
                    "Times Cited per Year",
                    "Avg Times Cited",
                    "H index",
                    "M index",
                    "G index",
                ],
            ),
            dash.top_n(),
            dash.dropdown(
                desc="Sort by:",
                options=[
                    "Num Documents",
                    "Frac Num Documents",
                    "Times Cited",
                    "Times Cited per Year",
                    "Avg Times Cited",
                    "H index",
                    "M index",
                    "G index",
                    "*Index*",
                ],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        # ----------------------------------------------------------------------
        if self.menu == "General":

            self.set_options(
                name="Column:",
                options=[
                    "Authors",
                    "Institutions",
                    "Institution_1st_Author",
                    "Countries",
                    "Country_1st_Author",
                    "Source_title",
                ],
            )

            self.set_options(
                name="View:",
                options=[
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                    "Pie plot",
                    "Wordcloud",
                    "Treemap",
                ],
            )

            self.set_options(name="Top by:", options=["Num Documents", "Times Cited",])

            self.set_options(
                name="Sort by:", options=["Alphabetic", "Num Documents", "Times Cited",]
            )

            if self.view == "Table":

                self.set_enabled("Top by:")
                self.set_enabled("Top N:")
                self.set_enabled("Sort by:")
                self.set_enabled("Ascending:")
                self.set_disabled("Colormap:")
                self.set_disabled("Width:")
                self.set_disabled("Height:")

            else:

                if self.view in ["Bar plot", "Horizontal bar plot"]:

                    self.set_enabled("Top by:")
                    self.set_enabled("Top N:")
                    self.set_enabled("Sort by:")
                    self.set_enabled("Ascending:")
                    self.set_enabled("Colormap:")
                    self.set_enabled("Width:")
                    self.set_enabled("Height:")

                else:

                    self.set_enabled("Top by:")
                    self.set_enabled("Top N:")
                    self.set_disabled("Sort by:")
                    self.set_disabled("Ascending:")
                    self.set_enabled("Colormap:")
                    self.set_enabled("Width:")
                    self.set_enabled("Height:")

        # ----------------------------------------------------------------------
        if self.menu == "Impact":

            COLUMNS = sorted(
                [column for column in self.data.columns if column not in EXCLUDE_COLS]
            )

            self.set_options(
                "Column:", options=[z for z in COLUMNS if z in self.data.columns]
            )

            self.set_options(
                "View:", options=["Table", "Bar plot", "Horizontal bar plot",]
            )

            self.set_options(
                "Top by:",
                options=[
                    "Num Documents",
                    "Times Cited",
                    "Times Cited per Year",
                    "Avg Times Cited",
                    "H index",
                    "M index",
                    "G index",
                ],
            )

            self.set_options(
                "Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Times Cited",
                    "Times Cited per Year",
                    "Avg Times Cited",
                    "H index",
                    "M index",
                    "G index",
                ],
            )

            if self.view == "Table":

                self.set_enabled("View:")
                self.set_enabled("Top by:")
                self.set_enabled("Top N:")
                self.set_enabled("Sort by:")
                self.set_enabled("Ascending:")
                self.set_disabled("Colormap:")
                self.set_disabled("Width:")
                self.set_disabled("Height:")

            else:

                self.set_enabled("Top by:")
                self.set_enabled("Top N:")
                self.set_enabled("Sort by:")
                self.set_enabled("Ascending:")
                self.set_enabled("Colormap:")
                self.set_enabled("Width:")
                self.set_enabled("Height:")

        # ----------------------------------------------------------------------
        if self.menu == "Single/Multiple publication":

            self.set_options(
                "Column:",
                options=[
                    "Authors",
                    "Institutions",
                    "Institution_1st_Author",
                    "Countries",
                    "Country_1st_Author",
                ],
            )

            self.set_options(
                "View:", options=["Table", "Bar plot", "Horizontal bar plot",]
            )

            self.set_options("Top by:", options=["Num Documents", "Times Cited",])

            self.set_options(
                "Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Times Cited",
                    "SD",
                    "MD",
                    "SMR",
                ],
            )

            if self.panel_widgets[1]["widget"].value == "Table":

                self.set_enabled("Top by:")
                self.set_enabled("Top N:")
                self.set_enabled("Sort by:")
                self.set_enabled("Ascending:")
                self.set_disabled("Colormap:")
                self.set_disabled("Width:")
                self.set_disabled("Height:")

            else:

                self.set_enabled("Top by:")
                self.set_enabled("Top N:")
                self.set_enabled("Sort by:")
                self.set_enabled("Ascending:")
                self.set_enabled("Colormap:")
                self.set_enabled("Width:")
                self.set_enabled("Height:")

        # ----------------------------------------------------------------------
        if self.menu == "Worldmap":

            self.set_options("Column:", options=["Countries", "Country_1st_Author",])

            self.set_options("Top by:", options=["Num Documents", "Times Cited",])

            self.set_disabled("View:")
            self.set_disabled("Top N:")
            self.set_disabled("Sort by:")
            self.set_disabled("Ascending:")
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        # ----------------------------------------------------------------------
        if self.menu in ["Core authors", "Core source titles", "Top documents"]:
            for i, _ in enumerate(self.panel_widgets):
                self.panel_widgets[i]["widget"].disabled = True


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()
