"""
Co-occurrence Analysis
==================================================================================================



"""
import warnings

import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd


import techminer.common as cmn
import techminer.dashboard as dash
from techminer.dashboard import DASH
from techminer.tfidf import TF_matrix
from techminer.params import EXCLUDE_COLS
from pyvis.network import Network

from techminer.heatmap import heatmap as heatmap_
from techminer.bubble_plot import bubble_plot
from techminer.core import limit_to_exclude


warnings.filterwarnings("ignore", category=UserWarning)

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

        if self.column == self.by:
            self.X_ = None
            return

        W = self.data[[self.column, self.by, "ID"]].dropna()
        A = TF_matrix(W, self.column)
        B = TF_matrix(W, self.by)

        if self.top_by == "Data":

            A = limit_to_exclude(
                data=A,
                axis=1,
                column=self.column,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )
            B = limit_to_exclude(
                data=B,
                axis=1,
                column=self.by,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )
            matrix = np.matmul(B.transpose().values, A.values)
            matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)

            # sort max values per column
            max_columns = matrix.sum(axis=0)
            max_columns = max_columns.sort_values(ascending=False)
            max_columns = max_columns.head(self.top_n).index

            max_index = matrix.sum(axis=1)
            max_index = max_index.sort_values(ascending=False)
            max_index = max_index.head(self.top_n).index

            matrix = matrix.loc[
                [t for t in matrix.index if t in max_index],
                [t for t in matrix.columns if t in max_columns],
            ]

            matrix = cmn.add_counters_to_axis(
                X=matrix, axis=1, data=self.data, column=self.column
            )
            matrix = cmn.add_counters_to_axis(
                X=matrix, axis=0, data=self.data, column=self.by
            )
            self.X_ = matrix

        if self.top_by in ["Num Documents", "Times Cited"]:

            A = limit_to_exclude(
                data=A,
                axis=1,
                column=self.column,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )

            A = cmn.add_counters_to_axis(
                X=A, axis=1, data=self.data, column=self.column
            )

            A = cmn.sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)

            A = A[A.columns[: self.top_n]]

            B = limit_to_exclude(
                data=B,
                axis=1,
                column=self.by,
                limit_to=self.limit_to,
                exclude=self.exclude,
            )

            B = cmn.add_counters_to_axis(X=B, axis=1, data=self.data, column=self.by)

            B = cmn.sort_by_axis(data=B, sort_by=self.top_by, ascending=False, axis=1)
            B = B[B.columns[: self.top_n]]

            matrix = np.matmul(B.transpose().values, A.values)
            matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)
            self.X_ = matrix

        self.sort()

    def sort(self):

        self.X_ = cmn.sort_by_axis(
            data=self.X_,
            sort_by=self.sort_r_axis_by,
            ascending=self.r_axis_ascending,
            axis=0,
        )

        self.X_ = cmn.sort_by_axis(
            data=self.X_,
            sort_by=self.sort_c_axis_by,
            ascending=self.c_axis_ascending,
            axis=1,
        )

    def matrix(self):
        self.apply()
        return self.X_.style.background_gradient(cmap=self.cmap, axis=None)

    def heatmap(self):
        self.apply()
        return heatmap_(self.X_, cmap=self.cmap, figsize=(self.width, self.height),)

    def bubble_plot(self):
        self.apply()
        return bubble_plot(
            self.X_, axis=0, cmap=self.cmap, figsize=(self.width, self.height),
        )

    def network_nx(self):
        ##
        self.apply()
        ##

        X = self.X_

        ## Networkx
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        G = nx.Graph(ax=ax)
        G.clear()

        ## Data preparation
        terms = X.columns.tolist() + X.index.tolist()

        node_sizes = cmn.counters_to_node_sizes(x=terms)
        column_node_sizes = node_sizes[: len(X.index)]
        index_node_sizes = node_sizes[len(X.index) :]

        node_colors = cmn.counters_to_node_colors(x=terms, cmap=lambda w: w)
        column_node_colors = node_colors[: len(X.index)]
        index_node_colors = node_colors[len(X.index) :]

        cmap = pyplot.cm.get_cmap(self.cmap)
        cmap_by = pyplot.cm.get_cmap(self.cmap_by)

        index_node_colors = [cmap_by(t) for t in index_node_colors]
        column_node_colors = [cmap(t) for t in column_node_colors]

        ## Add nodes
        G.add_nodes_from(terms)

        ## node positions
        if self.layout == "Spring":
            pos = nx.spring_layout(G, iterations=self.nx_iterations)
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
        m = X.stack().to_frame().reset_index()
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.0]
        m = m.reset_index(drop=True)

        max_width = m.link_.max()
        for idx in range(len(m)):

            edge = [(m.from_[idx], m.to_[idx])]
            width = 0.2 + 3.8 * m.link_[idx] / max_width
            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                node_size=1,
                with_labels=False,
                edge_color="k",
                edgelist=edge,
                width=width,
            )

        #
        # Draw column nodes
        #
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            edge_color="k",
            nodelist=X.columns.tolist(),
            node_size=column_node_sizes,
            node_color=column_node_colors,
            node_shape="o",
            edgecolors="k",
            linewidths=1,
        )

        #
        # Draw index nodes
        #
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            edge_color="k",
            nodelist=X.index.tolist(),
            node_size=index_node_sizes,
            node_color=index_node_colors,
            node_shape="o",
            edgecolors="k",
            linewidths=1,
        )

        node_sizes = column_node_sizes + index_node_sizes
        cmn.ax_text_node_labels(
            ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes
        )
        cmn.ax_expand_limits(ax)
        ax.set_aspect("equal")
        ax.axis("off")
        cmn.set_ax_splines_invisible(ax)
        return fig

    ##
    ##
    ## inferfaz con pyviz
    def network_interactive(self):
        ##
        self.apply()
        ##

        X = self.X_.copy()

        G = Network("700px", "870px", notebook=True)

        ## Data preparation
        terms = X.columns.tolist() + X.index.tolist()

        node_sizes = cmn.counters_to_node_sizes(x=terms)
        column_node_sizes = node_sizes[: len(X.index)]
        index_node_sizes = node_sizes[len(X.index) :]

        node_colors = cmn.counters_to_node_colors(x=terms, cmap=lambda w: w)
        column_node_colors = node_colors[: len(X.index)]
        index_node_colors = node_colors[len(X.index) :]

        cmap = pyplot.cm.get_cmap(self.cmap)
        cmap_by = pyplot.cm.get_cmap(self.cmap_by)
        column_node_colors = [cmap(t) for t in column_node_colors]
        column_node_colors = [
            matplotlib.colors.rgb2hex(t[:3]) for t in column_node_colors
        ]
        index_node_colors = [cmap_by(t) for t in index_node_colors]
        index_node_colors = [
            matplotlib.colors.rgb2hex(t[:3]) for t in index_node_colors
        ]

        for i_term, term in enumerate(X.columns.tolist()):
            G.add_node(
                term,
                size=column_node_sizes[i_term] / 100,
                color=column_node_colors[i_term],
            )

        for i_term, term in enumerate(X.index.tolist()):
            G.add_node(
                term,
                size=index_node_sizes[i_term] / 100,
                color=index_node_colors[i_term],
            )

        # links
        m = X.stack().to_frame().reset_index()
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.0]
        m = m.reset_index(drop=True)

        max_width = m.link_.max()
        for idx in range(len(m)):
            value = 0.5 + 2.5 * m.link_[idx] / max_width
            G.add_edge(
                m.from_[idx], m.to_[idx], width=value, color="lightgray", physics=False
            )

        return G.show("net.html")

    def slope_chart(self):
        ##
        self.apply()
        ##

        matrix = self.X_
        ##
        matplotlib.rc("font", size=12)

        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.cmap)
        cmap_by = pyplot.cm.get_cmap(self.cmap_by)

        m = len(matrix.index)
        n = len(matrix.columns)
        maxmn = max(m, n)
        yleft = (maxmn - m) / 2.0 + np.linspace(0, m, m)
        yright = (maxmn - n) / 2.0 + np.linspace(0, n, n)

        ax.vlines(
            x=1,
            ymin=-1,
            ymax=maxmn + 1,
            color="gray",
            alpha=0.7,
            linewidth=1,
            linestyles="dotted",
        )

        ax.vlines(
            x=3,
            ymin=-1,
            ymax=maxmn + 1,
            color="gray",
            alpha=0.7,
            linewidth=1,
            linestyles="dotted",
        )

        #
        # Dibuja los ejes para las conexiones
        #
        ax.scatter(x=[1] * m, y=yleft, s=1)
        ax.scatter(x=[3] * n, y=yright, s=1)

        #
        # Dibuja las conexiones
        #
        maxlink = matrix.max().max()
        minlink = matrix.values.ravel()
        minlink = min([v for v in minlink if v > 0])
        for idx, index in enumerate(matrix.index):
            for icol, col in enumerate(matrix.columns):
                link = matrix.loc[index, col]
                if link > 0:
                    ax.plot(
                        [1, 3],
                        [yleft[idx], yright[icol]],
                        c="k",
                        linewidth=0.5 + 4 * (link - minlink) / (maxlink - minlink),
                        alpha=0.5 + 0.5 * (link - minlink) / (maxlink - minlink),
                    )

        #
        # Sizes
        #
        left_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.index]
        right_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.columns]

        min_size = min(left_sizes + right_sizes)
        max_size = max(left_sizes + right_sizes)

        left_sizes = [
            150 + 2000 * (t - min_size) / (max_size - min_size) for t in left_sizes
        ]
        right_sizes = [
            150 + 2000 * (t - min_size) / (max_size - min_size) for t in right_sizes
        ]

        #
        # Colors
        #
        left_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.index]
        right_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.columns]

        min_color = min(left_colors + right_colors)
        max_color = max(left_colors + right_colors)

        left_colors = [
            cmap_by(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
            for t in left_colors
        ]
        right_colors = [
            cmap(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
            for t in right_colors
        ]

        ax.scatter(
            x=[1] * m,
            y=yleft,
            s=left_sizes,
            c=left_colors,
            zorder=10,
            linewidths=1,
            edgecolors="k",
        )

        for idx, text in enumerate(matrix.index):
            ax.plot([0.7, 1.0], [yleft[idx], yleft[idx]], "-", c="grey")

        for idx, text in enumerate(matrix.index):
            ax.text(
                0.7,
                yleft[idx],
                text,
                fontsize=10,
                ha="right",
                va="center",
                zorder=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
            )

        #
        # right y-axis
        #

        ax.scatter(
            x=[3] * n,
            y=yright,
            s=right_sizes,
            c=right_colors,
            zorder=10,
            linewidths=1,
            edgecolors="k",
        )

        for idx, text in enumerate(matrix.columns):
            ax.plot([3.0, 3.3], [yright[idx], yright[idx]], "-", c="grey")

        for idx, text in enumerate(matrix.columns):
            ax.text(
                3.3,
                yright[idx],
                text,
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                zorder=11,
            )

        #
        # Figure size
        #
        cmn.ax_expand_limits(ax)
        ax.invert_yaxis()
        ax.axis("off")

        return fig


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.app_title = "Bigraph Analysis"
        self.menu_options = [
            "Matrix",
            "Heatmap",
            "Bubble plot",
            "Network nx",
            "Network interactive",
            "Slope chart",
        ]

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.dropdown(
                desc="By:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(
                desc="Top by:", options=["Num Documents", "Times Cited", "Data",],
            ),
            dash.top_n(),
            dash.dropdown(
                desc="Sort C-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "Data",],
            ),
            dash.c_axis_ascending(),
            dash.dropdown(
                desc="Sort R-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "Data",],
            ),
            dash.r_axis_ascending(),
            dash.cmap(arg="cmap", desc="Colormap Col:"),
            dash.cmap(arg="cmap_by", desc="Colormap By:"),
            dash.nx_layout(),
            dash.nx_iterations(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.column == self.by:
            for i, _ in enumerate(self.panel_widgets[2:]):
                self.panel_widgets[i + 2]["widget"].disabled = True
            return
        else:
            for i, _ in enumerate(self.panel_widgets[2:]):
                self.panel_widgets[i + 2]["widget"].disabled = False

        if self.menu in ["Matrix", "Network interactive"]:
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu in ["Network nx", "Network interactive", "Slope chart"]:
            self.set_enabled("Colormap by:")
        else:
            self.set_disabled("Colormap by:")

        if self.menu == "Network nx":
            self.set_enabled("Layout:")
        else:
            self.set_disabled("Layout:")

        if self.menu == "Network nx" and self.layout == "Spring":
            self.set_enabled("nx interations:")
        else:
            self.set_disabled("nx iterations:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

