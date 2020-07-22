import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
import techminer.by_term as by_term
import techminer.common as common
import techminer.gui as gui
import techminer.plots as plt
from cdlib import algorithms
from ipywidgets import AppLayout, GridspecLayout, Layout
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from techminer.document_term import TF_matrix
from techminer.explode import __explode
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS

import techminer.common as cmn

from techminer.dashboard import DASH

###############################################################################
##
##  BASE FUNCTIONS
##
###############################################################################


def network_normalization(X, normalization=None):
    """
    """
    X = X.copy()
    if isinstance(normalization, str) and normalization == "None":
        normalization = None
    if normalization is None:
        X = X.applymap(lambda w: int(w))
    else:
        X = X.applymap(lambda w: float(w))
        M = X.copy()
    if normalization == "association":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / (M.loc[row, row] * M.at[col, col])
    if normalization == "inclusion":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / min(M.loc[row, row], M.at[col, col])
    if normalization == "jaccard":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / (
                    M.loc[row, row] + M.at[col, col] - M.at[row, col]
                )
    if normalization == "salton":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / np.sqrt(
                    (M.loc[row, row] * M.at[col, col])
                )
    return X


def co_occurrence_matrix(
    data,
    column,
    top_by,
    top_n,
    sort_c_axis_by,
    sort_r_axis_by,
    c_axis_ascending,
    r_axis_ascending,
    limit_to,
    exclude,
):
    """
    """

    W = data[[column, "ID"]].dropna()
    A = TF_matrix(W, column)
    A = cmn.limit_to_exclude(
        data=A, axis=1, column=column, limit_to=limit_to, exclude=exclude,
    )

    A = cmn.add_counters_to_axis(X=A, axis=1, data=data, column=column)

    A = cmn.sort_by_axis(data=A, sort_by=top_by, ascending=False, axis=1)

    A = A[A.columns[:top_n]]

    matrix = np.matmul(A.transpose().values, A.values)
    matrix = pd.DataFrame(matrix, columns=A.columns, index=A.columns)

    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_r_axis_by, ascending=r_axis_ascending, axis=0,
    )

    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_c_axis_by, ascending=c_axis_ascending, axis=1,
    )

    return matrix


def network_map(
    X, cmap, clustering, layout, only_communities, iterations, figsize=(8, 8)
):

    #
    # Network generation
    #
    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    terms = X.columns.tolist()
    n = len(terms)
    G.add_nodes_from(terms)

    max_width = 0
    for icol in range(n - 1):
        for irow in range(icol + 1, n):
            link = X.loc[X.columns[irow], X.columns[icol]]
            if link > 0:
                G.add_edge(terms[icol], terms[irow], width=link)
                if max_width < link:
                    max_width = link

    if clustering is None:
        cmap = pyplot.cm.get_cmap(cmap)
        node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in X.columns]
        max_citations = max(node_colors)
        min_citations = min(node_colors)
        node_colors = [
            cmap(0.2 + 0.80 * (i - min_citations) / (max_citations - min_citations))
            for i in node_colors
        ]

    if clustering in ["Label propagation", "Leiden", "Louvain", "Markov", "Walktrap"]:

        colors = []
        for cmap_name in ["tab20", "tab20b", "tab20c"]:
            cmap = pyplot.cm.get_cmap(cmap_name)
            colors += [cmap(0.025 + 0.05 * i) for i in range(20)]

        R = {
            "Label propagation": algorithms.label_propagation,
            "Leiden": algorithms.leiden,
            "Louvain": algorithms.louvain,
            "Walktrap": algorithms.walktrap,
        }[clustering](G).communities

        if only_communities:
            R = [sorted(r) for r in R]
            return R

        clusters = {}
        for idx, r in enumerate(R):
            for e in r:
                clusters[e] = idx
        node_colors = [colors[clusters[t]] for t in terms]

    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in X.columns]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    node_sizes = [
        600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
    ]

    if layout == "Spring":
        pos = nx.spring_layout(G, iterations=iterations)
    else:
        pos = {
            "Circular": nx.circular_layout,
            "Kamada Kawai": nx.kamada_kawai_layout,
            "Planar": nx.planar_layout,
            "Random": nx.random_layout,
            "Spectral": nx.spectral_layout,
            "Spring": nx.spring_layout,
            "Shell": nx.shell_layout,
        }[layout](G)

    for e in G.edges.data():
        a, b, width = e
        edge = [(a, b)]
        width = 0.2 + 4.0 * width["width"] / max_width
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=edge,
            width=width,
            edge_color="k",
            with_labels=False,
            node_size=1,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        edge_color="k",
        nodelist=terms,
        node_size=node_sizes,
        node_color=node_colors,
        node_shape="o",
        edgecolors="k",
        linewidths=1,
    )

    common.ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)

    fig.set_tight_layout(True)
    common.ax_expand_limits(ax)
    common.set_ax_splines_invisible(ax)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig


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
        self.X_ = None
        ##
        self.c_axis_ascending = None
        self.cmap = None
        self.column = None
        self.height = None
        self.layout = (None,)
        self.normalization = None
        self.r_axis_ascending = None
        self.sort_c_axis_by = None
        self.sort_r_axis_by = None
        self.top_by = None
        self.top_n = None
        self.width = None
        self.nx_iterations = None
        self.clustering = None

    def fit(self):
        self.X_ = co_occurrence_matrix(
            data=self.data,
            column=self.column,
            top_by=self.top_by,
            top_n=self.top_n,
            sort_c_axis_by=self.sort_c_axis_by,
            sort_r_axis_by=self.sort_r_axis_by,
            c_axis_ascending=self.c_axis_ascending,
            r_axis_ascending=self.r_axis_ascending,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        self.X_ = network_normalization(X=self.X_, normalization=self.normalization)

    def matrix(self):
        self.fit()
        if self.normalization == "None":
            return self.X_.style.background_gradient(cmap=self.cmap, axis=None)
        else:
            return self.X_.style.format("{:0.3f}").background_gradient(
                cmap=self.cmap, axis=None
            )

    def heatmap(self):
        self.fit()
        return plt.heatmap(self.X_, cmap=self.cmap, figsize=(self.width, self.height))

    def bubble_plot(self):
        self.fit()
        return plt.bubble(
            self.X_, axis=0, cmap=self.cmap, figsize=(self.width, self.height)
        )

    def network(self):
        self.fit()
        return network_map(
            self.X_,
            cmap=self.cmap,
            layout=self.layout,
            clustering=self.clustering,
            only_communities=False,
            iterations=self.nx_iterations,
            figsize=(self.width, self.height),
        )

    def communities(self):
        self.fit()
        return network_map(
            self.X_,
            cmap=self.cmap,
            layout=self.layout,
            clustering=self.clustering,
            only_communities=True,
            figsize=(self.width, self.height),
        )


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None):

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.data = data
        self.app_title = "Graph Analysis"
        self.menu_options = [
            "Matrix",
            "Heatmap",
            "Bubble plot",
            "Network",
            "Communities",
        ]

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_widgets = [
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            gui.top_n(),
            gui.dropdown(
                desc="Sort C-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "Data",],
            ),
            gui.c_axis_ascending(),
            gui.dropdown(
                desc="Sort R-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "Data",],
            ),
            gui.r_axis_ascending(),
            gui.normalization(),
            gui.dropdown(
                desc="Clustering:",
                options=["Label propagation", "Leiden", "Louvain", "Walktrap",],
            ),
            gui.cmap(),
            gui.nx_layout(),
            gui.nx_iterations(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Matrix":
            self.panel_widgets[-5]["widget"].disabled = True
            self.panel_widgets[-4]["widget"].disabled = False
            self.panel_widgets[-3]["widget"].disabled = True
            self.panel_widgets[-2]["widget"].disabled = True
            self.panel_widgets[-1]["widget"].disabled = True

        if self.menu in ["Heatmap", "Bubble plot"]:
            self.panel_widgets[-5]["widget"].disabled = False
            self.panel_widgets[-4]["widget"].disabled = False
            self.panel_widgets[-3]["widget"].disabled = True
            self.panel_widgets[-2]["widget"].disabled = False
            self.panel_widgets[-1]["widget"].disabled = False

        if self.menu == "Network":
            self.panel_widgets[-5]["widget"].disabled = False
            self.panel_widgets[-4]["widget"].disabled = False
            self.panel_widgets[-3]["widget"].disabled = False
            self.panel_widgets[-2]["widget"].disabled = False
            self.panel_widgets[-1]["widget"].disabled = False


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()

