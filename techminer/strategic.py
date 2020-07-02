"""
Strategic and thematic maps
==================================================================================================



"""

## path_length = nx.shortest_path_length(self._graph)
## distances = pd.DataFrame(index=self._graph.nodes(), columns=self._graph.nodes())
## for row, data in path_length:
##     for col, dist in data.items():
##         distances.loc[row, col] = dist
## distances = distances.fillna(distances.max().max())
## return nx.kamada_kawai_layout(self._graph, dist=distances.to_dict())

import json

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import techminer.by_term
import techminer.common as common
import techminer.graph as graph
import techminer.gui as gui
import techminer.plots as plt
from IPython.display import clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from techminer.document_term import document_term_matrix
from techminer.plots import COLORMAPS

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50


def strategic_map(
    X, n_clusters=2, linkage="ward", output=0, cmap="Greys", figsize=(6, 6),
):

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(1 - X)
    cluster_dict = {key: value for key, value in zip(X.columns, clustering.labels_)}

    map = pd.DataFrame(
        {"cluster": list(range(n_clusters))}, index=list(range(n_clusters))
    )
    map["density"] = 0.0
    map["centrality"] = 0.0
    map["name"] = ""
    map["n_members"] = 0
    map["members"] = [[]] * len(map)

    #
    # Members of cluster
    #
    for t in X.columns:
        map.at[cluster_dict[t], "members"] = map.loc[cluster_dict[t], "members"] + [t]

    #
    # Name of cluster
    #
    for i_cluster, words in enumerate(map["members"]):
        cluster_name = None
        cluster_freq = None
        map.at[i_cluster, "n_members"] = len(words)
        for word in words:
            freq = int(word.split(" ")[-1].split(":")[0])
            if cluster_freq is None or freq > cluster_freq:
                cluster_name = word
                cluster_freq = freq
        map.at[i_cluster, "name"] = cluster_name

    for i_cluster in range(len(map)):
        map.at[i_cluster, "members"] = ";".join(map.loc[i_cluster, "members"])

    #
    # density
    #
    for i_cluster in range(n_clusters):
        Z = X[[t for t in X.columns if cluster_dict[t] == i_cluster]]
        I = Z.copy()
        I = I.loc[[t for t in I.index if cluster_dict[t] == i_cluster], :]
        if len(I) == 1:
            map.at[i_cluster, "density"] = 0
        else:
            density = []
            for i_column in range(len(I.columns) - 1):
                for i_index in range(i_column + 1, len(I.index)):
                    density.append(I.loc[I.index[i_index], I.columns[i_column]])

            map.at[i_cluster, "density"] = sum(density) / len(density)

    #
    # centratity
    #
    for i_cluster in range(n_clusters):
        Z = X[[t for t in X.columns if cluster_dict[t] == i_cluster]]
        I = Z.copy()
        I = I.loc[[t for t in I.index if cluster_dict[t] != i_cluster], :]
        map.at[i_cluster, "centrality"] = I.sum().sum()

    #
    # Output
    #
    if output == 0:
        return map

    if output == 1:
        text = {}
        for i_cluster in range(n_clusters):
            text[map.name[i_cluster]] = map.members[i_cluster].split(";")
        return json.dumps(text, indent=4, sort_keys=True)

    if output == 2:

        matplotlib.rc("font", size=11)
        cmap = pyplot.cm.get_cmap(cmap)
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()

        node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in map.name]
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        node_sizes = [
            600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

        node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in map.name]
        max_citations = max(node_colors)
        min_citations = min(node_colors)
        node_colors = [
            cmap(0.2 + 0.80 * (i - min_citations) / (max_citations - min_citations))
            for i in node_colors
        ]

        ax.scatter(
            map.centrality,
            map.density,
            s=node_sizes,
            linewidths=1,
            edgecolors="k",
            c=node_colors,
        )

        common.ax_expand_limits(ax)
        common.ax_node_labels(
            ax, labels=map.name, x=map.centrality, y=map.density, node_sizes=node_sizes
        )

        ax.axhline(
            y=map.density.median(),
            color="gray",
            linestyle="--",
            linewidth=0.5,
            zorder=-1,
        )
        ax.axvline(
            x=map.centrality.median(),
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

        #  ax.set_aspect("equal")
        ax.axis("off")

        common.set_ax_splines_invisible(ax)

        fig.set_tight_layout(True)

        return fig

    return None


MAP_COLUMNS = [
    "Author_Keywords",
    "Author_Keywords_CL",
    "Index_Keywords",
    "Index_Keywords_CL",
    "Abstract_words",
    "Title_words",
]


def __TAB0__(data, limit_to, exclude):
    #
    # UI
    #
    output = widgets.Output()
    #
    COLUMNS = [column for column in data.columns if column in MAP_COLUMNS]
    #
    left_panel = [
        gui.dropdown(desc="View:", options=["Table", "Membership", "Plot"],),
        gui.dropdown(
            desc="Column:", options=[z for z in COLUMNS if z in data.columns],
        ),
        gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
        gui.top_n(m=10, n=1001, i=10),
        gui.normalization(),
        gui.n_clusters(),
        gui.linkage(),
        gui.cmap(),
        gui.fig_width(),
        gui.fig_height(),
    ]
    #
    # Server
    #
    def server(**kwargs):
        #
        # Logic
        #
        view = kwargs["view"]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        normalization = kwargs["normalization"]
        n_clusters = int(kwargs["n_clusters"])
        linkage = kwargs["linkage"]
        cmap = kwargs["cmap"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        matrix = graph.co_occurrence_matrix(
            data=data,
            column=column,
            top_by=top_by,
            top_n=top_n,
            normalization=normalization,
            limit_to=limit_to,
            exclude=exclude,
        )

        output.clear_output()
        with output:
            if view == "Table":
                display(
                    strategic_map(
                        X=matrix,
                        n_clusters=n_clusters,
                        linkage=linkage,
                        output=0,
                        cmap=cmap,
                        figsize=(width, height),
                    )
                )

            if view == "Membership":
                display(
                    print(
                        strategic_map(
                            X=matrix,
                            n_clusters=n_clusters,
                            linkage=linkage,
                            cmap=cmap,
                            output=1,
                            figsize=(width, height),
                        )
                    )
                )

            if view == "Plot":
                display(
                    strategic_map(
                        X=matrix,
                        n_clusters=n_clusters,
                        linkage=linkage,
                        output=2,
                        cmap=cmap,
                        figsize=(width, height),
                    )
                )

        return

    #
    # Body
    #
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


def _get_fmt(summ):
    n_Num_Documents = int(np.log10(summ["Num_Documents"].max())) + 1
    n_Times_Cited = int(np.log10(summ["Times_Cited"].max())) + 1
    return "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"


###############################################################################
##
##  APP
##
###############################################################################


def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Strategic Map",
        tab_titles=["Strategic Map",],
        tab_widgets=[__TAB0__(data, limit_to=limit_to, exclude=exclude),],
        tab=tab,
    )
