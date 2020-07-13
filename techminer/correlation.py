"""
Correlation Analysis
==================================================================================================


"""

import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
import techminer.by_term as by_term
import techminer.common as cmn
import techminer.gui as gui
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from techminer.chord_diagram import ChordDiagram
from techminer.graph import co_occurrence_matrix
from techminer.bigraph import filter_index
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.keywords import Keywords
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS
from techminer.document_term import TF_matrix

from matplotlib.lines import Line2D

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
        ##
        self.ascending = None
        self.by = None
        self.c_axis_ascending = None
        self.cmap = None
        self.column = None
        self.height = None
        self.iterations = None
        self.layout = None
        self.method = None
        self.r_axis_ascending = None
        self.sort_c_axis_by = None
        self.sort_r_axis_by = None
        self.top_by = None
        self.top_n = None
        self.width = None
        ##

    def fit(self):

        x = self.data.copy()

        if self.column == self.by:

            A = TF_matrix(x, column=self.column)
            A = cmn.limit_to_exclude(
                data=A,
                axis=1,
                column=self.column,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )
            A = cmn.add_counters_to_axis(X=A, axis=1, data=x, column=self.column)
            A = cmn.sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)
            A = A[A.columns[: self.top_n]]
            matrix = A.corr(method=self.method)

        else:

            w = x[[self.column, self.by, "ID"]].dropna()

            A = TF_matrix(w, column=self.column)
            A = cmn.limit_to_exclude(
                data=A,
                axis=1,
                column=self.column,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )
            A = cmn.add_counters_to_axis(X=A, axis=1, data=x, column=self.column)
            A = cmn.sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)
            A = A[A.columns[: self.top_n]]

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
        self.fit()
        return self.X_.style.format("{:+4.3f}").background_gradient(
            cmap=self.cmap, axis=None
        )

    def heatmap(self):
        self.fit()
        return plt.heatmap(self.X_, cmap=self.cmap, figsize=(self.width, self.height))

    def bubble_plot(self):
        self.fit()
        return plt.bubble(
            self.X_, axis=0, cmap=self.cmap, figsize=(self.width, self.height),
        )

    def chord_diagram(self):
        self.fit()
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

    def correlation_map(self):
        self.fit()

        if len(self.X_.columns) > 50:
            return "Maximum number of nodes exceded!"

        ## Networkx
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        G = nx.Graph(ax=ax)
        G.clear()

        ## Data preparation
        terms = self.X_.columns.tolist()
        node_sizes = cmn.counters_to_node_sizes(x=terms)
        node_colors = cmn.counters_to_node_colors(
            x=terms, cmap=pyplot.cm.get_cmap(self.cmap)
        )

        ## Add nodes
        G.add_nodes_from(terms)

        ## node positions
        if self.layout == "Spring":
            pos = nx.spring_layout(G, iterations=self.iterations)
        else:
            pos = {
                "Circular": nx.circular_layout,
                "Kamada Kawai": nx.kamada_kawai_layout,
                "Planar": nx.planar_layout,
                "Random": nx.random_layout,
                "Spectral": nx.spectral_layout,
                "Spring": nx.spring_layout,
                "Shell": nx.shell_layout,
            }[self.layout](G)

        ## links
        m = self.X_.stack().to_frame().reset_index()
        m = m[m.level_0 < m.level_1]
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.0]
        m = m.reset_index(drop=True)

        d = {
            0: {"width": 4, "style": "solid", "edge_color": "k"},
            1: {"width": 2, "style": "solid", "edge_color": "k"},
            2: {"width": 1, "style": "dashed", "edge_color": "gray"},
            3: {"width": 1, "style": "dotted", "edge_color": "gray"},
        }

        n_edges_0 = 0
        n_edges_25 = 0
        n_edges_50 = 0
        n_edges_75 = 0

        for idx in range(len(m)):

            edge = [(m.from_[idx], m.to_[idx])]
            key = (
                0
                if m.link_[idx] > 0.75
                else (1 if m.link_[idx] > 0.50 else (2 if m.link_[idx] > 0.25 else 3))
            )

            n_edges_75 += 1 if m.link_[idx] >= 0.75 else 0
            n_edges_50 += 1 if m.link_[idx] >= 0.50 and m.link_[idx] < 0.75 else 0
            n_edges_25 += 1 if m.link_[idx] >= 0.25 and m.link_[idx] < 0.50 else 0
            n_edges_0 += 1 if m.link_[idx] > 0 and m.link_[idx] < 0.25 else 0

            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                node_size=1,
                with_labels=False,
                edgelist=edge,
                **(d[key])
            )

        ## nodes
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            ax=ax,
            edge_color="k",
            nodelist=terms,
            node_size=node_sizes,
            node_color=node_colors,
            node_shape="o",
            edgecolors="k",
            linewidths=1,
        )

        ## node labels
        cmn.ax_text_node_labels(
            ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes
        )

        ## Figure size
        cmn.ax_expand_limits(ax)

        ##
        legend_lines = [
            Line2D([0], [0], color="k", linewidth=4, linestyle="-"),
            Line2D([0], [0], color="k", linewidth=2, linestyle="-"),
            Line2D([0], [0], color="gray", linewidth=1, linestyle="--"),
            Line2D([0], [0], color="gray", linewidth=1, linestyle=":"),
        ]

        text_75 = "> 0.75 ({})".format(n_edges_75)
        text_50 = "0.50-0.75 ({})".format(n_edges_50)
        text_25 = "0.25-0.50 ({})".format(n_edges_25)
        text_0 = "< 0.25 ({})".format(n_edges_0)

        ax.legend(legend_lines, [text_75, text_50, text_25, text_0])

        ax.axis("off")

        return fig


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
    def __init__(self, data, limit_to=None, exclude=None):
        """Dashboard app"""

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.app_title = "Correlation Analysis"
        self.menu_options = [
            "Matrix",
            "Heatmap",
            "Bubble plot",
            "Correlation map",
            "Chord diagram",
        ]

        self.panel_widgets = [
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(
                desc="By:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(desc="Method:", options=["pearson", "kendall", "spearman"],),
            gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            gui.top_n(),
            gui.dropdown(
                desc="Sort C-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            gui.c_axis_ascending(),
            gui.dropdown(
                desc="Sort R-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            gui.r_axis_ascending(),
            gui.cmap(),
            gui.nx_layout(),
            gui.nx_iterations(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        for i in [-9, -8, -7, -6]:
            self.panel_widgets[i]["widget"].disabled = (
                True
                if self.menu not in ["Matrix", "Heatmap", "Bubble plot",]
                else False
            )

        self.panel_widgets[-4]["widget"].disabled = self.menu != "Correlation map"
        self.panel_widgets[-3]["widget"].disabled = (
            False
            if self.menu == "Correlation map" and self.layout == "Spring"
            else True
        )

        self.panel_widgets[-2]["widget"].disabled = self.menu == "Matrix"
        self.panel_widgets[-1]["widget"].disabled = self.menu == "Matrix"


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()

