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

import techminer.common as common

# from networkx.algorithms.community.label_propagation import (
#     label_propagation_communities,
# )

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50


def _get_fmt(summ):
    n_Num_Documents = int(np.log10(summ["Num_Documents"].max())) + 1
    n_Times_Cited = int(np.log10(summ["Times_Cited"].max())) + 1
    return "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"


def _get_num_documents(x):
    z = x.split(" ")[-1]
    z = z.split(":")
    return z[0] + z[1] + x


def _get_times_cited(x):
    z = x.split(" ")[-1]
    z = z.split(":")
    return z[1] + z[0] + x


def _normalize(X, normalization=None):
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
    top_by=0,
    top_n=5,
    normalization=None,
    sort_by="Alphabetic",
    ascending=True,
    limit_to=None,
    exclude=None,
):
    """
    """

    def get_top_terms(summ, top_by, top_n, limit_to, exclude):

        summ = summ.copy()

        if isinstance(top_by, str):
            top_by = top_by.replace(" ", "_")
            top_by = {"Num_Documents": 0, "Times_Cited": 1,}[top_by]

        if top_by == 0:
            summ = summ.sort_values(
                ["Num_Documents", "Times_Cited"], ascending=[False, False],
            )

        if top_by == 1:
            summ = summ.sort_values(
                ["Times_Cited", "Num_Documents"], ascending=[False, False, True],
            )

        if isinstance(limit_to, dict):
            if column in limit_to.keys():
                limit_to = limit_to[column]
            else:
                limit_to = None

        if limit_to is not None:
            summ = summ[summ.index.map(lambda w: w in limit_to)]

        if isinstance(exclude, dict):
            if column in exclude.keys():
                exclude = exclude[column]
            else:
                exclude = None

        if exclude is not None:
            summ = summ[summ.index.map(lambda w: w not in exclude)]

        summ = summ.head(top_n)

        return summ.index.tolist()

    #
    # main body
    #
    summ = by_term.analytics(data, column)
    fmt = _get_fmt(summ)
    new_names = {
        key: fmt.format(key, nd, tc)
        for key, nd, tc in zip(summ.index, summ.Num_Documents, summ.Times_Cited)
    }
    top_terms = get_top_terms(
        summ, top_by=top_by, top_n=top_n, limit_to=limit_to, exclude=exclude
    )

    W = data[[column, "ID"]].dropna()
    A = TF_matrix(W, column)
    A = A[[t for t in A.columns if t in top_terms]]
    matrix = np.matmul(A.transpose().values, A.values)
    matrix = pd.DataFrame(matrix, columns=A.columns, index=A.columns)
    matrix.columns = [new_names[t] for t in matrix.columns]
    matrix.index = [new_names[t] for t in matrix.index]

    matrix = _normalize(matrix, normalization)

    if sort_by == "Alphabetic":
        matrix = matrix.sort_index(axis=0, ascending=ascending)

    if sort_by == "Num Documents":
        terms = matrix.index.tolist()
        terms = sorted(terms, reverse=not ascending, key=_get_num_documents)
        matrix = matrix.loc[terms, terms]

    if sort_by == "Times Cited":
        terms = matrix.index.tolist()
        terms = sorted(terms, reverse=not ascending, key=_get_times_cited)
        matrix = matrix.loc[terms, terms]

    return matrix


def network_map(X, cmap, clustering, layout, only_communities, figsize=(8, 8)):

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


def __TAB0__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        # 0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Matrix", "Heatmap", "Bubble plot", "Network", "Communities"],
                layout=Layout(width="55%"),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="55%"),
            ),
        },
        # 2
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Times Cited",], layout=Layout(width="55%"),
            ),
        },
        # 3
        gui.top_n(),
        # 4
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=["Alphabetic", "Num Documents", "Times Cited",],
                layout=Layout(width="55%"),
            ),
        },
        # 5
        gui.ascending(),
        gui.normalization(),
        # 7
        {
            "arg": "clustering",
            "desc": "Clustering:",
            "widget": widgets.Dropdown(
                options=["None", "Label propagation", "Leiden", "Louvain", "Walktrap",],
                layout=Layout(width="55%"),
            ),
        },
        # 8
        gui.cmap(),
        gui.nx_layout(),
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
        view = kwargs["view"]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        sort_by = kwargs["sort_by"]
        ascending = kwargs["ascending"]
        normalization = kwargs["normalization"]
        clustering = None if kwargs["clustering"] == "None" else kwargs["clustering"]
        cmap = kwargs["cmap"]
        layout = kwargs["layout"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        if view == "Matrix":
            left_panel[-5]["widget"].disabled = True
            left_panel[-4]["widget"].disabled = False
            left_panel[-3]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = True
            left_panel[-1]["widget"].disabled = True

        if view in ["Heatmap", "Bubble plot"]:
            left_panel[-5]["widget"].disabled = True
            left_panel[-4]["widget"].disabled = False
            left_panel[-3]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = False
            left_panel[-1]["widget"].disabled = False

        if view == "Network":
            left_panel[-5]["widget"].disabled = False
            left_panel[-4]["widget"].disabled = False
            left_panel[-3]["widget"].disabled = False
            left_panel[-2]["widget"].disabled = False
            left_panel[-1]["widget"].disabled = False

        matrix = co_occurrence_matrix(
            data=data,
            column=column,
            top_by=top_by,
            top_n=top_n,
            normalization=normalization,
            sort_by=sort_by,
            ascending=ascending,
            limit_to=limit_to,
            exclude=exclude,
        )

        output.clear_output()
        with output:
            if view == "Matrix":
                if normalization == "None":
                    display(matrix.style.background_gradient(cmap=cmap, axis=None))
                else:
                    display(
                        matrix.style.format("{:0.3f}").background_gradient(
                            cmap=cmap, axis=None
                        )
                    )

            if view == "Heatmap":
                display(plt.heatmap(matrix, cmap=cmap, figsize=(width, height)))

            if view == "Bubble plot":
                display(plt.bubble(matrix, axis=0, cmap=cmap, figsize=(width, height)))

            if view in ["Network", "Communities"]:
                only_communities = True if view == "Communities" else False
                display(
                    network_map(
                        matrix,
                        cmap=cmap,
                        layout=layout,
                        clustering=clustering,
                        only_communities=only_communities,
                        figsize=(width, height),
                    )
                )

        return

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


def associations_map(X, selected, cmap, layout, figsize):

    if selected == ():
        return "There are not associations to show"

    #
    # Network generation
    #
    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    Y = X[[t for t in X.columns if t in selected]]
    S = Y.sum(axis=1)
    S = S[S > 0]
    X = Y.loc[S.index, :]
    if len(X) == 0:
        return "There are not associations to show"

    terms = X.index.tolist()
    G.add_nodes_from(terms)

    max_width = 0
    for icol in range(len(X.columns)):
        for irow in range(len(X.index)):
            if X.index[irow] != X.columns[icol]:
                link = X.loc[X.index[irow], X.columns[icol]]
                if link > 0:
                    G.add_edge(X.index[irow], X.columns[icol], width=link)
                    if max_width < link:
                        max_width = link

    cmap = pyplot.cm.get_cmap(cmap)
    node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in X.index]
    max_citations = max(node_colors)
    min_citations = min(node_colors)
    if max_citations == min_citations:
        node_colors = [cmap(0.9)] * len(X.index)
    else:
        node_colors = [
            cmap(0.2 + (t - min_citations) / (max_citations - min_citations))
            for t in node_colors
        ]

    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in X.index]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    if max_size == min_size:
        node_sizes = [600] * len(node_sizes)
    else:
        node_sizes = [
            600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

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
    common.ax_expand_limits(ax)
    common.set_ax_splines_invisible(ax)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.set_tight_layout(True)

    return fig


#
# Association Map
#
def __TAB1__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        # 0
        {
            "arg": "column",
            "desc": "Column:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="55%"),
            ),
        },
        # 1
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Times Cited",], layout=Layout(width="55%"),
            ),
        },
        # 2
        gui.top_n(),
        # 3
        gui.normalization(),
        gui.cmap(),
        # 5
        gui.nx_layout(),
        # 6
        {
            "arg": "width",
            "desc": "Width:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), ensure_option=True, layout=Layout(width="55%"),
            ),
        },
        # 7
        {
            "arg": "height",
            "desc": "Height:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), ensure_option=True, layout=Layout(width="55%"),
            ),
        },
        # 8
        {
            "arg": "selected",
            "desc": "Seleted Cols:",
            "widget": widgets.widgets.SelectMultiple(
                options=[], layout=Layout(width="95%", height="212px"),
            ),
        },
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
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        layout = kwargs["layout"]
        normalization = kwargs["normalization"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        selected = kwargs["selected"]

        matrix = co_occurrence_matrix(
            data=data,
            column=column,
            top_by=top_by,
            top_n=top_n,
            normalization=normalization,
            limit_to=limit_to,
            exclude=exclude,
        )

        left_panel[-1]["widget"].options = sorted(matrix.columns)

        output.clear_output()
        with output:
            display(
                associations_map(
                    X=matrix,
                    selected=selected,
                    cmap=cmap,
                    layout=layout,
                    figsize=(width, height),
                )
            )
        #
        return

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


#
#
#
#
#


def association_analysis(
    X,
    method="MDS",
    n_components=2,
    n_clusters=2,
    linkage="ward",
    x_axis=0,
    y_axis=1,
    figsize=(6, 6),
):

    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(1 - X)
    cluster_dict = {key: value for key, value in zip(X.columns, clustering.labels_)}

    if method == "MDS":
        # Multidimensional scaling
        embedding = MDS(n_components=n_components)
        X_transformed = embedding.fit_transform(X,)

    if method == "CA":
        # Correspondence analysis
        X_transformed = correspondence_matrix(X)

    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = pyplot.cm.get_cmap(cmap_name)
        colors += [cmap(0.025 + 0.05 * i) for i in range(20)]

    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in X.columns]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    node_sizes = [
        600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
    ]

    node_colors = [
        cmap(0.2 + 0.80 * cluster_dict[t] / (n_clusters - 1)) for t in X.columns
    ]

    x_axis = X_transformed[:, x_axis]
    y_axis = X_transformed[:, y_axis]

    ax.scatter(
        x_axis, y_axis, s=node_sizes, linewidths=1, edgecolors="k", c=node_colors
    )

    common.ax_expand_limits(ax)

    pos = {term: (x_axis[idx], y_axis[idx]) for idx, term in enumerate(X.columns)}
    common.ax_text_node_labels(
        ax=ax, labels=X.columns, dict_pos=pos, node_sizes=node_sizes
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)

    ax.set_aspect("equal")
    ax.axis("off")
    common.set_ax_splines_invisible(ax)

    fig.set_tight_layout(True)

    return fig


def correspondence_matrix(X):
    """
    """

    matrix = X.values
    grand_total = np.sum(matrix)
    correspondence_matrix = np.divide(matrix, grand_total)
    row_totals = np.sum(correspondence_matrix, axis=1)
    col_totals = np.sum(correspondence_matrix, axis=0)
    independence_model = np.outer(row_totals, col_totals)
    norm_correspondence_matrix = np.divide(correspondence_matrix, row_totals[:, None])
    distances = np.zeros(
        (correspondence_matrix.shape[0], correspondence_matrix.shape[0])
    )
    norm_col_totals = np.sum(norm_correspondence_matrix, axis=0)
    for row in range(correspondence_matrix.shape[0]):
        distances[row] = np.sqrt(
            np.sum(
                np.square(norm_correspondence_matrix - norm_correspondence_matrix[row])
                / col_totals,
                axis=1,
            )
        )
    std_residuals = np.divide(
        (correspondence_matrix - independence_model), np.sqrt(independence_model)
    )
    u, s, vh = np.linalg.svd(std_residuals, full_matrices=False)
    deltaR = np.diag(np.divide(1.0, np.sqrt(row_totals)))
    rowScores = np.dot(np.dot(deltaR, u), np.diag(s))

    return rowScores


#
# Association analysis
#
def __TAB2__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        # 0
        {
            "arg": "method",
            "desc": "Method:",
            "widget": widgets.Dropdown(
                options=["Multidimensional scaling", "Correspondence analysis"],
                layout=Layout(width="55%"),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="55%"),
            ),
        },
        # 2
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Times Cited",], layout=Layout(width="55%"),
            ),
        },
        # 3
        gui.top_n(),
        gui.normalization(),
        gui.n_components(),
        gui.n_clusters(),
        gui.linkage(),
        gui.x_axis(),
        gui.y_axis(),
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
        method = {"Multidimensional scaling": "MDS", "Correspondence analysis": "CA"}[
            kwargs["method"]
        ]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        normalization = kwargs["normalization"]
        n_components = int(kwargs["n_components"])
        n_clusters = int(kwargs["n_clusters"])
        x_axis = int(kwargs["x_axis"])
        y_axis = int(kwargs["y_axis"])
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        left_panel[8]["widget"].options = list(range(n_components))
        left_panel[9]["widget"].options = list(range(n_components))
        x_axis = left_panel[8]["widget"].value
        y_axis = left_panel[9]["widget"].value

        matrix = co_occurrence_matrix(
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

            display(
                association_analysis(
                    X=matrix,
                    method=method,
                    n_components=n_components,
                    n_clusters=n_clusters,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    figsize=(width, height),
                )
            )

        return

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


###############################################################################
##
##  APP
##
###############################################################################


def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Graph Analysis",
        tab_titles=["Network Map", "Associations Map", "Association analysis",],
        tab_widgets=[
            __TAB0__(data, limit_to=limit_to, exclude=exclude),
            __TAB1__(data, limit_to=limit_to, exclude=exclude),
            __TAB2__(data, limit_to=limit_to, exclude=exclude),
        ],
        tab=tab,
    )
