import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms
from pyvis.network import Network
import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.dashboard import DASH
from techminer.document_term import TF_matrix
from techminer.params import EXCLUDE_COLS

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

    #
    # 1.-- Computes TF_matrix with occurrence >= min_occurrence
    #
    W = data[[column, "ID"]].dropna()
    A = TF_matrix(data=W, column=column, scheme=None, min_occurrence=1)
    A = cmn.limit_to_exclude(
        data=A, axis=1, column=column, limit_to=limit_to, exclude=exclude,
    )

    #
    # 2.-- Select top_n
    #
    A = cmn.add_counters_to_axis(X=A, axis=1, data=data, column=column)
    A = cmn.sort_by_axis(data=A, sort_by=top_by, ascending=False, axis=1)
    A = A[A.columns[:top_n]]

    #
    # 4.-- computes co-occurrence
    #
    matrix = np.matmul(A.transpose().values, A.values)
    matrix = pd.DataFrame(matrix, columns=A.columns, index=A.columns)

    #
    # 5.-- Matrix sort
    #
    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_r_axis_by, ascending=r_axis_ascending, axis=0,
    )
    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_c_axis_by, ascending=c_axis_ascending, axis=1,
    )

    return matrix


def network_map_nx(
    X, cmap, clustering, layout, only_communities, iterations, n_labels, figsize=(8, 8)
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
        300 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
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

    cmn.ax_text_node_labels(
        ax=ax, labels=terms[0:n_labels], dict_pos=pos, node_sizes=node_sizes
    )

    fig.set_tight_layout(True)
    cmn.ax_expand_limits(ax)
    cmn.set_ax_splines_invisible(ax)
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
        self.clustering = None
        self.cmap = None
        self.column = None
        self.height = None
        self.layout = None
        self.max_nodes = None
        self.normalization = None
        self.nx_iterations = None
        self.r_axis_ascending = None
        self.sort_c_axis_by = None
        self.sort_r_axis_by = None
        self.top_by = None
        self.top_n = None
        self.width = None


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):

        DASH.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )

        self.app_title = "Graph Analysis"
        self.menu_options = [
            "Matrix",
            "Heatmap",
            "Bubble plot",
            "Network (nx)",
            "Network (interactive)",
            "Communities",
        ]

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.normalization(),
            dash.dropdown(
                desc="Clustering:",
                options=["Label propagation", "Leiden", "Louvain", "Walktrap",],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
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

        if self.menu == "Matrix":

            self.set_disabled("Clustering:")
            self.set_disabled("Colormap:")
            self.set_disabled("Layout:")
            self.set_disabled("nx iterations:")
            self.set_disabled("N labels:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu in ["Heatmap", "Bubble plot"]:

            self.set_disabled("Clustering:")
            self.set_enabled("Colormap:")
            self.set_disabled("Layout:")
            self.set_disabled("nx iterations:")
            self.set_disabled("N labels:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu == "Network (nx)" or self.menu == "Communities":

            self.set_enabled("Clustering:")
            self.set_enabled("Colormap:")
            self.set_enabled("Layout:")
            self.set_enabled("N labels:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

            if self.menu == "Network" and self.layout == "Spring":
                self.set_enabled("nx iterations:")
            else:
                self.set_disabled("nx iterations:")

        if self.menu == "Network (interactive)":

            self.set_enabled("Clustering:")
            self.set_disabled("Colormap:")
            self.set_disabled("Layout:")
            self.set_disabled("N labels:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")
            self.set_disabled("nx iterations:")

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

    def network_nx(self):
        self.fit()
        self.X_ = cmn.sort_by_axis(
            data=self.X_, sort_by=self.top_by, ascending=False, axis=0
        )
        self.X_ = cmn.sort_by_axis(
            data=self.X_, sort_by=self.top_by, ascending=False, axis=1
        )
        return network_map_nx(
            self.X_,
            cmap=self.cmap,
            layout=self.layout,
            clustering=self.clustering,
            only_communities=False,
            n_labels=self.n_labels,
            iterations=self.nx_iterations,
            figsize=(self.width, self.height),
        )

    def communities(self):
        self.fit()
        return network_map_nx(
            self.X_,
            cmap=self.cmap,
            layout=self.layout,
            clustering=self.clustering,
            only_communities=True,
            iterations=self.nx_iterations,
            n_labels=None,
            figsize=(self.width, self.height),
        )

    def network_interactive(self):
        ##
        self.fit()
        ##

        X = self.X_.copy()
        G = nx.Graph()

        # Network generation
        terms = X.columns.tolist()
        n = len(terms)
        G.add_nodes_from(terms)

        max_width = X.max().max()
        m = X.stack().to_frame().reset_index()
        m = m[m.level_0 < m.level_1]
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.001]
        m = m.reset_index(drop=True)

        for idx in range(len(m)):
            value = 0.1 + 1.4 * m.link_[idx] / max_width
            G.add_edge(
                m.from_[idx], m.to_[idx], width=value, color="lightgray", physics=False
            )

        R = {
            "Label propagation": algorithms.label_propagation,
            "Leiden": algorithms.leiden,
            "Louvain": algorithms.louvain,
            "Walktrap": algorithms.walktrap,
        }[self.clustering](G).communities

        for i_community, community in enumerate(R):
            for item in community:
                G.nodes[item]["group"] = i_community

        node_sizes = cmn.counters_to_node_sizes(terms)
        node_sizes = [size / 100 for size in node_sizes]

        for i_term, term in enumerate(terms):
            G.nodes[term]["size"] = node_sizes[i_term]

        nt = Network("700px", "870px", notebook=True)
        nt.from_nx(G)

        return nt.show("net.html")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()
