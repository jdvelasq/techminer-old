import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib
import techminer.common as cmn

import techminer.core.dashboard as dash
from techminer.core import DASH


from techminer.chord_diagram import ChordDiagram

from techminer.tfidf import TF_matrix
from techminer.network import Network
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

        x = self.data.copy()

        if self.column == self.by:

            #
            # 1.-- Drop NA from column
            #
            w = x[[self.column, "ID"]].dropna()

            #
            # 2.-- Computes TF_matrix with occurrence >= min_occurrence
            #
            A = TF_matrix(
                data=w,
                column=self.column,
                scheme=None,
                min_occurrence=self.min_occurrence,
            )

            #
            # 3.-- Limit to/Exclude
            #
            A = limit_to_exclude(
                data=A,
                axis=1,
                column=self.column,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )

            #
            # 4.-- Select max_items
            #
            A = cmn.add_counters_to_axis(
                X=A, axis=1, data=self.data, column=self.column
            )

            A = cmn.sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)

            A = A[A.columns[: self.max_items]]

            if len(A.columns) > self.max_items:
                top_items = A.sum(axis=0)
                top_items = top_items.sort_values(ascending=False)
                top_items = top_items.head(self.max_items)
                A = A.loc[:, top_items.index]
                rows = A.sum(axis=1)
                rows = rows[rows > 0]
                A = A.loc[rows.index, :]

            #
            # 5.-- Computes correlation
            #
            matrix = A.corr(method=self.method)

        else:

            #
            # 1.-- Drop NA from column
            #
            w = x[[self.column, self.by, "ID"]].dropna()

            #
            # 2.-- Computes TF_matrix with occurrence >= min_occurrence
            #
            A = TF_matrix(data=w, column=self.column, scheme=None, min_occurrence=None,)

            #
            # 3.-- Limit to/Exclude
            #
            A = limit_to_exclude(
                data=A,
                axis=1,
                column=self.column,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )

            #
            # 4.-- Minimal occurrence
            #
            terms = A.sum(axis=0)
            terms = terms.sort_values(ascending=False)
            terms = terms[terms >= self.min_occurrence]
            A = A.loc[:, terms.index]

            #
            # 5.-- Select max_items
            #
            A = cmn.add_counters_to_axis(
                X=A, axis=1, data=self.data, column=self.column
            )

            A = cmn.sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)

            if len(A.columns) > self.max_items:
                A = A[A.columns[: self.max_items]]

            #
            # 6.-- Computes correlation
            #
            B = TF_matrix(w, column=self.by)
            matrix = np.matmul(B.transpose().values, A.values)
            matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)
            matrix = matrix.corr(method=self.method)

        matrix = cmn.sort_by_axis(
            data=matrix,
            sort_by=self.sort_r_axis_by,
            ascending=self.r_axis_ascending,
            axis=0,
        )

        matrix = cmn.sort_by_axis(
            data=matrix,
            sort_by=self.sort_c_axis_by,
            ascending=self.c_axis_ascending,
            axis=1,
        )
        self.X_ = matrix

    def matrix(self):
        self.apply()
        return self.X_.style.format("{:+4.3f}").background_gradient(
            cmap=self.cmap, axis=None
        )

    def heatmap(self):
        self.apply()
        return plt.heatmap(self.X_, cmap=self.cmap, figsize=(self.width, self.height))

    def bubble_plot(self):
        self.apply()
        return plt.bubble(
            self.X_, axis=0, cmap=self.cmap, figsize=(self.width, self.height),
        )

    def chord_diagram(self):
        self.apply()
        x = self.X_.copy()
        terms = self.X_.columns.tolist()
        node_sizes = cmn.counters_to_node_sizes(x=terms)
        node_colors = cmn.counters_to_node_colors(x, cmap=pyplot.cm.get_cmap(self.cmap))

        cd = ChordDiagram()

        ## add nodes
        for idx, term in enumerate(x.columns):
            cd.add_node(term, color=node_colors[idx], s=node_sizes[idx])

        ## add links
        m = x.stack().to_frame().reset_index()
        m = m[m.level_0 < m.level_1]
        m.columns = ["from_", "to_", "link_"]
        m = m.reset_index(drop=True)

        d = {
            0: {"linestyle": "-", "linewidth": 4, "color": "black"},
            1: {"linestyle": "-", "linewidth": 2, "color": "black"},
            2: {"linestyle": "--", "linewidth": 1, "color": "gray"},
            3: {"linestyle": ":", "linewidth": 1, "color": "lightgray"},
        }

        for idx in range(len(m)):

            key = (
                0
                if m.link_[idx] > 0.75
                else (1 if m.link_[idx] > 0.50 else (2 if m.link_[idx] > 0.25 else 3))
            )

            cd.add_edge(m.from_[idx], m.to_[idx], **(d[key]))

        return cd.plot(figsize=(self.width, self.height))

    def correlation_map_nx(self):
        self.apply()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).networkx_plot(
            layout=self.layout,
            iterations=self.nx_iterations,
            figsize=(self.width, self.height),
        )

    def communities(self):
        self.fit()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).cluster_members_

    def correlation_map_interactive(self):
        self.apply()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).pyvis_plot()


###############################################################################
##
##  DASHBOARD
##
###############################################################################

COLUMNS = sorted(
    [
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
)


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.app_title = "Correlation Analysis"
        self.menu_options = [
            "Matrix",
            "Heatmap",
            "Bubble plot",
            "Correlation map (nx)",
            "Correlation map (interactive)",
            "Chord diagram",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.dropdown(
                desc="By:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.dropdown(desc="Method:", options=["pearson", "kendall", "spearman"],),
            dash.min_occurrence(),
            dash.max_items(),
            dash.dropdown(
                desc="Clustering:",
                options=["Label propagation", "Leiden", "Louvain", "Walktrap",],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            dash.dropdown(
                desc="Sort C-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            dash.c_axis_ascending(),
            dash.dropdown(
                desc="Sort R-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            dash.r_axis_ascending(),
            dash.cmap(),
            dash.nx_layout(),
            dash.n_labels(),
            dash.nx_iterations(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu in [
            "Matrix",
            "Heatmap",
            "Bubble plot",
        ]:
            self.set_enabled("Sort C-axis by:")
            self.set_enabled("C-axis ascending:")
            self.set_enabled("Sort R-axis by:")
            self.set_enabled("R-axis ascending:")
        else:
            self.set_disabled("Sort C-axis by:")
            self.set_disabled("C-axis ascending:")
            self.set_disabled("Sort R-axis by:")
            self.set_disabled("R-axis ascending:")

        if self.menu == "Correlation map (nx)":
            self.set_enabled("Layout:")
            self.set_enabled("N labels:")
        else:
            self.set_disabled("Layout:")
            self.set_disabled("N labels:")

        if self.menu == "Correlation map" and self.layout == "Spring":
            self.set_enabled("nx iterations:")
        else:
            self.set_disabled("nx iterations:")

        if self.menu in ["Matrix", "Correlation map (interactive)"]:
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
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

