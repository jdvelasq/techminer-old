"""
Analysis by Term
==========================================================================

"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

import techminer.plots as plt
from techminer.explode import __explode as _explode
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS
import techminer.common as cmn
import techminer.gui as gui
from techminer.dashboard import DASH

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

    def fit(self):
        x = self.data.copy()
        last_year = x.Year.max()

        # SD: single document / MD: multiple document
        x["SD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
        )
        x["MD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
        )

        x["Num_Documents"] = 1
        x["Frac_Num_Documents"] = x[self.column].map(
            lambda w: round(1 / (len(w.split(";")) if isinstance(w, str) else 1), 2)
            if not pd.isna(w)
            else 0
        )
        x["First_Year"] = x.Year
        x = _explode(
            x[
                [
                    self.column,
                    "Num_Documents",
                    "Times_Cited",
                    "Frac_Num_Documents",
                    "First_Year",
                    "SD",
                    "MD",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=False).agg(
            {
                "Num_Documents": np.sum,
                "Times_Cited": np.sum,
                "Frac_Num_Documents": np.sum,
                "First_Year": np.min,
                "SD": np.sum,
                "MD": np.sum,
            }
        )
        result["Last_Year"] = last_year
        result = result.assign(
            ID=x.groupby(self.column).agg({"ID": list}).reset_index()["ID"]
        )
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

        result["SMR"] = [
            round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)
        ]

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

        #
        # Orden de las columnas
        #
        result = result.reset_index(drop=True)
        result = result.set_index(self.column)

        result = result[
            [
                "Num_Documents",
                "Frac_Num_Documents",
                "Times_Cited",
                "Times_Cited_per_Year",
                "Avg_Times_Cited",
                "H_index",
                "M_index",
                "G_index",
                "SD",
                "MD",
                "SMR",
                "First_Year",
                "Last_Year",
                "Years",
                "ID",
            ]
        ]

        #
        # Limit to
        #
        limit_to = self.limit_to
        if isinstance(limit_to, dict):
            if self.column in limit_to.keys():
                limit_to = limit_to[self.column]
            else:
                limit_to = None

        if limit_to is not None:
            index = [w for w in result.index if w in limit_to]
            result = result.loc[index, :]

        #
        # Exclude
        #
        exclude = self.exclude
        if isinstance(exclude, dict):
            if self.column in exclude.keys():
                exclude = exclude[self.column]
            else:
                exclude = None

        if exclude is not None:
            index = [w for w in result.index if w not in exclude]
            result = result.loc[index, :]

        #
        # Top by
        #
        if isinstance(self.top_by, str):
            top_by = self.top_by.replace(" ", "_")
            top_by = {
                "Num_Documents": 0,
                "Times_Cited": 1,
                "Frac_Num_Documents": 2,
                "Times_Cited_per_Year": 3,
                "Avg_Times_Cited": 4,
                "H_index": 5,
                "M_index": 6,
                "G_index": 7,
            }[top_by]

        if top_by is not None:

            by, ascending_top_by = {
                0: (["Num_Documents", "Times_Cited"], False),
                1: (["Times_Cited", "Frac_Num_Documents"], False),
                2: (["Frac_Num_Documents", "Times_Cited"], False),
                3: (["Times_Cited_per_Year", "Frac_Num_Documents"], False,),
                4: (["Avg_Times_Cited", "Frac_Num_Documents"], False,),
                5: (["H_index", "G_index", "Times_Cited"], False,),
                6: (["M_index", "G_index", "Times_Cited"], False,),
                7: (["G_index", "H_index", "Times_Cited"], False,),
            }[top_by]

            result.sort_values(
                by=by, ascending=ascending_top_by, inplace=True,
            )

        else:

            result.sort_values(
                ["Num_Documents", "Times_Cited"], ascending=False, inplace=True,
            )

        if self.top_n is not None:
            result = result.head(self.top_n)

        #
        #  sort_by
        #
        if isinstance(self.sort_by, str):
            sort_by = self.sort_by.replace(" ", "_")
            sort_by = {
                "Num_Documents": 0,
                "Frac_Num_Documents": 1,
                "Times_Cited": 2,
                "Times_Cited_per_Year": 3,
                "Avg_Times_Cited": 4,
                "H_index": 5,
                "M_index": 6,
                "G_index": 7,
                "*Index*": 8,
            }[sort_by]

        if sort_by == 8:
            result = result.sort_index(axis=0, ascending=self.ascending)
        else:
            sort_by = {
                0: ["Num_Documents", "Times_Cited", "H_index"],
                1: ["Frac_Num_Documents", "Times_Cited", "H_index"],
                2: ["Times_Cited", "Num_Documents", "H_index"],
                3: ["Times_Cited_per_Year", "Num_Documents", "Times_Cited"],
                4: ["Avg_Times_Cited", "Num_Documents", "Times_Cited"],
                5: ["H_index", "G_index", "Times_Cited", "Num_Documents"],
                6: ["M_index", "G_index", "Times_Cited", "Num_Documents"],
                7: ["G_index", "H_index", "Times_Cited", "Num_Documents"],
            }[sort_by]
            result = result.sort_values(by=sort_by, ascending=self.ascending)

        self.X_ = result

    def terms(self):
        self.fit()
        selected_columns = [
            "Num_Documents",
            "Times_Cited",
            "Times_Cited_per_Year",
            "Avg_Times_Cited",
        ]
        if self.column in [
            "Authors",
            "Countries",
            "Country_1st_Author",
            "Institutions",
            "Institution_1st_Author",
            "Abb_Source_Title",
        ]:
            selected_columns += ["H_index", "G_index", "M_index"]

        if self.column in ["Authors", "Countries", "Institutions"]:
            selected_columns += [
                "Frac_Num_Documents",
                "SD",
                "MD",
                "SMR",
            ]

        result = self.X_[selected_columns]

        return result

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

        ## Sort by
        if self.sort_by in result.columns:
            result = result.sort_values(self.sort_by, ascending=self.ascending)
        else:
            result = cmn.sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )

        ## counters in axis names
        result = result.set_index("Authors")
        result = cmn.add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
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
                xlabel=self.top_y,
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
        ]
        self.panel_widgets = [
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(
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
            gui.dropdown(
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
            gui.top_n(),
            gui.dropdown(
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
            gui.ascending(),
            gui.cmap(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        # ----------------------------------------------------------------------
        if self.menu == "General":
            self.panel_widgets[0]["widget"].options = [
                "Authors",
                "Institutions",
                "Institution_1st_Author",
                "Countries",
                "Country_1st_Author",
                "Institutions",
                "Institution_1st_Author",
                "Source_title",
            ]
            self.panel_widgets[1]["widget"].options = [
                "Table",
                "Bar plot",
                "Horizontal bar plot",
                "Pie plot",
                "Wordcloud",
                "Treemap",
            ]
            self.panel_widgets[2]["widget"].options = [
                "Num Documents",
                "Times Cited",
            ]
            self.panel_widgets[4]["widget"].options = [
                "Alphabetic",
                "Num Documents",
                "Times Cited",
            ]
            for i, _ in enumerate(self.panel_widgets):
                self.panel_widgets[i]["widget"].disabled = False

            if self.panel_widgets[1]["widget"].value == "Table":
                self.panel_widgets[-3]["widget"].disabled = True
                self.panel_widgets[-2]["widget"].disabled = True
                self.panel_widgets[-1]["widget"].disabled = True
            else:
                self.panel_widgets[-3]["widget"].disabled = False
                self.panel_widgets[-2]["widget"].disabled = False
                self.panel_widgets[-1]["widget"].disabled = False

        # ----------------------------------------------------------------------
        if self.menu == "Impact":
            self.panel_widgets[0]["widget"].options = [
                z for z in COLUMNS if z in data.columns
            ]
            self.panel_widgets[1]["widget"].options = [
                "Table",
                "Bar plot",
                "Horizontal bar plot",
            ]
            self.panel_widgets[2]["widget"].options = [
                "Num Documents",
                "Times Cited",
                "Times Cited per Year",
                "Avg Times Cited",
                "H index",
                "M index",
                "G index",
            ]
            self.panel_widgets[4]["widget"].options = [
                "Alphabetic",
                "Num Documents",
                "Times Cited",
                "Times Cited per Year",
                "Avg Times Cited",
                "H index",
                "M index",
                "G index",
            ]
            for i, _ in enumerate(self.panel_widgets):
                self.panel_widgets[i]["widget"].disabled = False

            if self.panel_widgets[1]["widget"].value == "Table":
                self.panel_widgets[-3]["widget"].disabled = True
                self.panel_widgets[-2]["widget"].disabled = True
                self.panel_widgets[-1]["widget"].disabled = True
            else:
                self.panel_widgets[-3]["widget"].disabled = False
                self.panel_widgets[-2]["widget"].disabled = False
                self.panel_widgets[-1]["widget"].disabled = False

        # ----------------------------------------------------------------------
        if self.menu == "Single/Multiple publication":
            self.panel_widgets[0]["widget"].options = [
                "Authors",
                "Institutions",
                "Institution_1st_Author",
                "Countries",
                "Country_1st_Author",
                "Institutions",
                "Institution_1st_Author",
            ]
            self.panel_widgets[1]["widget"].options = [
                "Table",
                "Bar plot",
                "Horizontal bar plot",
            ]
            self.panel_widgets[2]["widget"].options = [
                "Num Documents",
                "Times Cited",
            ]
            self.panel_widgets[4]["widget"].options = [
                "Alphabetic",
                "Num Documents",
                "Times Cited",
                "SD",
                "MD",
                "SMR",
            ]
            for i, _ in enumerate(self.panel_widgets):
                self.panel_widgets[i]["widget"].disabled = False

            if self.panel_widgets[1]["widget"].value == "Table":
                self.panel_widgets[-3]["widget"].disabled = True
                self.panel_widgets[-2]["widget"].disabled = True
                self.panel_widgets[-1]["widget"].disabled = True
            else:
                self.panel_widgets[-3]["widget"].disabled = False
                self.panel_widgets[-2]["widget"].disabled = False
                self.panel_widgets[-1]["widget"].disabled = False

        # ----------------------------------------------------------------------
        if self.menu == "Worldmap":
            self.panel_widgets[0]["widget"].options = [
                "Countries",
                "Country_1st_Author",
            ]
            self.panel_widgets[2]["widget"].options = [
                "Num Documents",
                "Times Cited",
            ]
            self.panel_widgets[0]["widget"].disabled = False
            self.panel_widgets[1]["widget"].disabled = True
            self.panel_widgets[2]["widget"].disabled = False
            self.panel_widgets[3]["widget"].disabled = True
            self.panel_widgets[4]["widget"].disabled = True
            self.panel_widgets[5]["widget"].disabled = True
            self.panel_widgets[6]["widget"].disabled = False
            self.panel_widgets[7]["widget"].disabled = False
            self.panel_widgets[8]["widget"].disabled = False

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


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #


def analytics(
    data,
    column,
    output=0,
    top_by=None,
    top_n=None,
    sort_by="Num_Documents",
    ascending=True,
    cmap="Greys",
    limit_to=None,
    exclude=None,
    fontsize=11,
    figsize=(6, 6),
):
    """Summarize the number of documents and citations by term in a dataframe.

    Args:
        x (pandas.DataFrame): Bibliographic dataframe
        column (str): Column to Analyze.
        limit_to (list): Limit the result to the terms in the list.
        exclude (list): Terms to be excluded.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Times_Cited": list(range(10,14)),
    ...          "Year": [1990, 1990, 1991, 1991],
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Times_Cited  Year  ID
    0  author 0;author 1;author 2           10  1990   0
    1                    author 0           11  1990   1
    2                    author 1           12  1991   2
    3                    author 3           13  1991   3

    >>> analytics(x, 'Authors')[['Num_Documents', 'Times_Cited']]
              Num_Documents  Times_Cited
    Authors                             
    author 2              1           10
    author 3              1           13
    author 0              2           21
    author 1              2           22

    >>> items = ['author 1', 'author 2']
    >>> analytics(x, 'Authors', limit_to=items)[['Num_Documents', 'Times_Cited']]
              Num_Documents  Times_Cited
    Authors                             
    author 2              1           10
    author 1              2           22
    

    >>> analytics(x, 'Authors', exclude=items)[['Num_Documents', 'Times_Cited']]
              Num_Documents  Times_Cited
    Authors                             
    author 3              1           13
    author 0              2           21
    

    """

    #
    # Computation
    #
    x = data.copy()
    last_year = x.Year.max()

    # SD: single document / MD: multiple document
    x["SD"] = x[column].map(
        lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
    )
    x["MD"] = x[column].map(
        lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
    )

    x["Num_Documents"] = 1
    x["Frac_Num_Documents"] = x[column].map(
        lambda w: round(1 / (len(w.split(";")) if isinstance(w, str) else 1), 2)
        if not pd.isna(w)
        else 0
    )
    x["First_Year"] = x.Year
    x = __explode(
        x[
            [
                column,
                "Num_Documents",
                "Times_Cited",
                "Frac_Num_Documents",
                "First_Year",
                "SD",
                "MD",
                "ID",
            ]
        ],
        column,
    )
    result = x.groupby(column, as_index=False).agg(
        {
            "Num_Documents": np.sum,
            "Times_Cited": np.sum,
            "Frac_Num_Documents": np.sum,
            "First_Year": np.min,
            "SD": np.sum,
            "MD": np.sum,
        }
    )
    result["Last_Year"] = last_year
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result = result.assign(Years=result.Last_Year - result.First_Year + 1)
    result = result.assign(Times_Cited_per_Year=result.Times_Cited / result.Years)
    result["Times_Cited_per_Year"] = result["Times_Cited_per_Year"].map(
        lambda w: round(w, 2)
    )
    result = result.assign(Avg_Times_Cited=result.Times_Cited / result.Num_Documents)
    result["Avg_Times_Cited"] = result["Avg_Times_Cited"].map(lambda w: round(w, 2))

    result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

    result["SMR"] = [round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)]

    #
    # Indice H
    #
    z = x[[column, "Times_Cited", "ID"]].copy()
    z = (
        x.assign(
            rn=x.sort_values("Times_Cited", ascending=False).groupby(column).cumcount()
            + 1
        )
    ).sort_values([column, "Times_Cited", "rn"], ascending=[False, False, True])
    z["rn2"] = z.rn.map(lambda w: w * w)

    q = z.query("Times_Cited >= rn")
    q = q.groupby(column, as_index=False).agg({"rn": np.max})
    h_dict = {key: value for key, value in zip(q[column], q.rn)}

    result["H_index"] = result[column].map(
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
    q = q.groupby(column, as_index=False).agg({"rn": np.max})
    h_dict = {key: value for key, value in zip(q[column], q.rn)}
    result["G_index"] = result[column].map(
        lambda w: h_dict[w] if w in h_dict.keys() else 0
    )

    #
    # Orden de las columnas
    #
    result = result.reset_index(drop=True)
    result = result.set_index(column)

    result = result[
        [
            "Num_Documents",
            "Frac_Num_Documents",
            "Times_Cited",
            "Times_Cited_per_Year",
            "Avg_Times_Cited",
            "H_index",
            "M_index",
            "G_index",
            "SD",
            "MD",
            "SMR",
            "First_Year",
            "Last_Year",
            "Years",
            "ID",
        ]
    ]

    #
    # Limit to
    #
    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        index = [w for w in result.index if w in limit_to]
        result = result.loc[index, :]

    #
    # Exclude
    #
    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        index = [w for w in result.index if w not in exclude]
        result = result.loc[index, :]

    #
    # Top by
    #
    if isinstance(top_by, str):
        top_by = top_by.replace(" ", "_")
        top_by = {
            "Num_Documents": 0,
            "Times_Cited": 1,
            "Frac_Num_Documents": 2,
            "Times_Cited_per_Year": 3,
            "Avg_Times_Cited": 4,
            "H_index": 5,
            "M_index": 6,
            "G_index": 7,
        }[top_by]

    if top_by is not None:

        by, ascending_top_by = {
            0: (["Num_Documents", "Times_Cited"], False),
            1: (["Times_Cited", "Frac_Num_Documents"], False),
            2: (["Frac_Num_Documents", "Times_Cited"], False),
            3: (["Times_Cited_per_Year", "Frac_Num_Documents"], False,),
            4: (["Avg_Times_Cited", "Frac_Num_Documents"], False,),
            5: (["H_index", "G_index", "Times_Cited"], False,),
            6: (["M_index", "G_index", "Times_Cited"], False,),
            7: (["G_index", "H_index", "Times_Cited"], False,),
        }[top_by]

        result.sort_values(
            by=by, ascending=ascending_top_by, inplace=True,
        )

    else:

        result.sort_values(
            ["Num_Documents", "Times_Cited"], ascending=False, inplace=True,
        )

    if top_n is not None:
        result = result.head(top_n)

    #
    #  sort_by
    #
    if isinstance(sort_by, str):
        sort_by = sort_by.replace(" ", "_")
        sort_by = {
            "Num_Documents": 0,
            "Frac_Num_Documents": 1,
            "Times_Cited": 2,
            "Times_Cited_per_Year": 3,
            "Avg_Times_Cited": 4,
            "H_index": 5,
            "M_index": 6,
            "G_index": 7,
            "*Index*": 8,
        }[sort_by]

    if sort_by == 8:
        result = result.sort_index(axis=0, ascending=ascending)
    else:
        sort_by = {
            0: ["Num_Documents", "Times_Cited", "H_index"],
            1: ["Frac_Num_Documents", "Times_Cited", "H_index"],
            2: ["Times_Cited", "Num_Documents", "H_index"],
            3: ["Times_Cited_per_Year", "Num_Documents", "Times_Cited"],
            4: ["Avg_Times_Cited", "Num_Documents", "Times_Cited"],
            5: ["H_index", "G_index", "Times_Cited", "Num_Documents"],
            6: ["M_index", "G_index", "Times_Cited", "Num_Documents"],
            7: ["G_index", "H_index", "Times_Cited", "Num_Documents"],
        }[sort_by]
        result = result.sort_values(by=sort_by, ascending=ascending)

    #
    #
    # Output
    #
    #
    if isinstance(output, str):
        output = output.replace(" ", "_")
        output = {
            "Analytics": 0,
            "Bar_plot": 1,
            "Horizontal_bar_plot": 2,
            "Pie_plot": 3,
            "Wordcloud": 4,
            "Treemap": 5,
            "S/D_Ratio_(bar)": 6,
            "S/D_Ratio_(barh)": 7,
        }[output]

    if output == 0:
        result.pop("ID")
        return result
        #  if cmap is None:
        #      return result
        # return result.style.background_gradient(cmap=cmap, axis=0)

    values, darkness = {
        0: ("Num_Documents", "Times_Cited"),
        1: ("Times_Cited", "Num_Documents"),
        2: ("Frac_Num_Documents", "Times_Cited"),
        3: ("Times_Cited_per_Year", "Num_Documents"),
        4: ("Avg_Times_Cited", "Num_Documents"),
        5: ("H_index", "Avg_Times_Cited"),
        6: ("M_index", "Avg_Times_Cited"),
        7: ("G_index", "Avg_Times_Cited"),
    }[top_by]

    if output == 1:
        return plt.bar(
            height=result[values],
            darkness=result[darkness],
            cmap=cmap,
            figsize=figsize,
        )

    if output == 2:
        return plt.barh(
            width=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 3:
        return plt.pie(
            x=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 4:
        return plt.wordcloud(
            x=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 5:
        return plt.treemap(
            x=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 6:
        z = result[["SD", "MD"]]
        return plt.stacked_bar(X=z, cmap=cmap, figsize=figsize,)

    if output == 7:
        z = result[["SD", "MD"]]
        return plt.stacked_barh(X=z, cmap=cmap, figsize=figsize,)

    return "ERROR: Output code unknown"


###############################################################################
##
##  APP
##
###############################################################################


def __TAB0__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        gui.dropdown(
            desc="View:",
            options=[
                "Analytics",
                "Bar plot",
                "Horizontal bar plot",
                "Pie plot",
                "Wordcloud",
                "Treemap",
                "S/D Ratio (bar)",
                "S/D Ratio (barh)",
            ],
        ),
        gui.dropdown(
            desc="Column:", options=[z for z in COLUMNS if z in data.columns],
        ),
        gui.dropdown(
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
        gui.top_n(),
        gui.dropdown(
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
        gui.ascending(),
        gui.cmap(),
        gui.fig_width(),
        gui.fig_height(),
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        output.clear_output()
        with output:
            display(widgets.HTML("Processing ..."))
        #
        view = kwargs["view"]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = kwargs["top_n"]
        cmap = kwargs["cmap"]
        sort_by = kwargs["sort_by"]
        ascending = kwargs["ascending"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        if view == "Analytics":
            left_panel[-1]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = True
        else:
            left_panel[-1]["widget"].disabled = False
            left_panel[-2]["widget"].disabled = False

        out = analytics(
            data=data,
            column=column,
            output=view,
            top_by=top_by,
            top_n=top_n,
            sort_by=sort_by,
            ascending=ascending,
            cmap=cmap,
            limit_to=limit_to,
            exclude=exclude,
            figsize=(width, height),
        )

        output.clear_output()
        with output:
            if view == "Analytics":
                display(out.style.background_gradient(cmap=cmap, axis=0))
            else:
                display(out)

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


#
#
#
#
def __TAB1__(data):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    left_panel = [
        gui.dropdown(desc="Column:", options=["Countries", "Country_1st_Author"],),
        gui.dropdown(desc="Top by:", options=["Num_Documents", "Times_Cited"],),
        gui.cmap(),
        gui.fig_width(),
        gui.fig_height(),
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
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        cmap = kwargs["cmap"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        #
        result = analytics(
            data, column=column, output=0, top_by="Num_Documents", top_n=None,
        )
        #
        output.clear_output()
        with output:
            display(plt.worldmap(x=result[top_by], figsize=(width, height), cmap=cmap,))

    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)

