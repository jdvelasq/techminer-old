"""
Co-occurrence Analysis
==================================================================================================



"""
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
import techminer.by_term as by_term
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.keywords import Keywords
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def filter_index(
    x,
    column,
    matrix,
    axis,
    top_by=0,
    top_n=10,
    sort_by=0,
    ascending=True,
    limit_to=None,
    exclude=None,
):
    """
    Args:
        x: bibliographic database
        column: column analyzed
        matrix: result
        axis: 0=rows, 1=columns
        top_by: 0=Num_Documents, 1=Times_Cited
        sort_by: 0=Alphabetic, 1=Num_Documents, 2=Times_Cited

    """
    top_terms = by_term.summary(x, column)

    if isinstance(top_by, str):
        top_by = top_by.replace(" ", "_")
        top_by = {"Num_Documents": 0, "Times_Cited": 1,}[top_by]

    if top_by == 0:
        top_terms = top_terms.sort_values(
            ["Num_Documents", "Times_Cited", column],
            ascending=[False, False, True],
            ignore_index=True,
        )

    if top_by == 1:
        top_terms = top_terms.sort_values(
            ["Times_Cited", "Num_Documents", column],
            ascending=[False, False, True],
            ignore_index=True,
        )

    #  top_terms = top_terms[column]

    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        top_terms = top_terms[top_terms[column].map(lambda w: w in limit_to)]

    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        top_terms = top_terms[top_terms[column].map(lambda w: w not in exclude)]

    if top_n is not None:
        top_terms = top_terms.head(top_n)

    if isinstance(sort_by, str):
        sort_by = sort_by.replace(" ", "_")
        sort_by = {"Alphabetic": 0, "Num_Documents": 1, "Times_Cited": 2,}[sort_by]

    if sort_by == 0:
        top_terms = top_terms.sort_values(column, ascending=ascending)

    if sort_by == 1:
        top_terms = top_terms.sort_values(
            ["Num_Documents", "Times_Cited"], ascending=ascending
        )

    if sort_by == 2:
        top_terms = top_terms.sort_values(
            ["Times_Cited", "Num_Documents"], ascending=ascending
        )

    top_terms = top_terms[column].tolist()

    if axis == 0 or axis == 2:
        matrix = matrix.loc[top_terms, :]

    if axis == 1 or axis == 2:
        matrix = matrix.loc[:, top_terms]

    return matrix


def document_term_matrix(x, column):
    """Computes the term-frequency matrix for the terms in a column.

    Args:
        column (str): the column to explode.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author_Keywords': y,
    ...       "Times_Cited": list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author_Keywords  Times_Cited  ID
    0       A               a            0   0
    1     A;B             a;b            1   1
    2       B               b            2   2
    3   A;B;C               c            3   3
    4     B;D             c;d            4   4

    >>> document_term_matrix(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1

    >>> document_term_matrix(df, 'Author_Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1



    """
    data = x[[column, "ID"]].copy()
    data["value"] = 1.0
    data = __explode(data, column)
    result = pd.pivot_table(
        data=data, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)
    return result


def co_occurrence(
    x, column, by=None, top_by=None, top_n=None, limit_to=None, exclude=None
):
    """Summary occurrence and citations by terms in two different columns.

    Args:
        by (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the by.
        column (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author_Keywords': y,
    ...       "Times_Cited": list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author_Keywords  Times_Cited  ID
    0       A               a            0   0
    1     A;B             a;b            1   1
    2       B               b            2   2
    3   A;B;C               c            3   3
    4     B;D             c;d            4   4

    >>> co_occurrence(df, column='Author_Keywords', by='Authors')
       a  b  c  d
    A  2  1  1  0
    B  1  2  2  1
    C  0  0  1  0
    D  0  0  1  1

    >>> co_occurrence(df, column='Author_Keywords', by='Authors', top_by='Frequency', top_n=3)
       a  b  c
    A  2  1  1
    B  1  2  2
    C  0  0  1

    >>> terms = ['A', 'B', 'c', 'd']
    >>> co_occurrence(df, column='Author_Keywords', by='Authors', limit_to=terms)
       c  d
    A  1  0
    B  2  1

    >>> co_occurrence(df, column='Author_Keywords', by='Authors', exclude=terms)
       a  b
    C  0  0
    D  0  0

    """
    x = x.copy()
    if by is None or by == column:
        by = column + "_"
        x[by] = x[column].copy()
        if isinstance(limit_to, dict) and column in limit_to.keys():
            limit_to[by] = limit_to[column]
        if isinstance(exclude, dict) and column in exclude.keys():
            exclude[by] = exclude[column]

    w = x[[column, by, "ID"]].dropna()
    A = document_term_matrix(w, column)
    A = filter_index(
        x=x,
        column=column,
        matrix=A,
        axis=1,
        top_by=top_by,
        top_n=top_n,
        limit_to=limit_to,
        exclude=exclude,
    )
    B = document_term_matrix(w, by)
    B = filter_index(
        x=x,
        column=by,
        matrix=B,
        axis=1,
        top_by=top_by,
        top_n=top_n,
        limit_to=limit_to,
        exclude=exclude,
    )

    result = np.matmul(B.transpose().values, A.values)
    result = pd.DataFrame(result, columns=A.columns, index=B.columns)
    result = result.sort_index(axis=0, ascending=True).sort_index(
        axis=1, ascending=True
    )

    return result


from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS


def occurrence(
    x,
    column,
    normalization=None,
    linkage="ward",
    n_clusters=3,
    n_components=2,
    n_dim=1,
    view="network",
    top_by="Num Documents",
    top_n=50,
    limit_to=None,
    exclude=None,
    layout="Kamada Kawai",
    cmap="Greys",
    figsize=(17, 12),
):
    """
    """

    def normalize(matrix, normalization=None):
        """
        """
        matrix = matrix.applymap(lambda w: float(w))
        m = matrix.copy()
        if normalization == "association":
            for col in m.columns:
                for row in m.index:
                    matrix.at[row, col] = m.at[row, col] / (
                        m.loc[row, row] * m.at[col, col]
                    )
        if normalization == "inclusion":
            for col in m.columns:
                for row in m.index:
                    matrix.at[row, col] = m.at[row, col] / min(
                        m.loc[row, row], m.at[col, col]
                    )
        if normalization == "jaccard":
            for col in m.columns:
                for row in m.index:
                    matrix.at[row, col] = m.at[row, col] / (
                        m.loc[row, row] + m.at[col, col] - m.at[row, col]
                    )
        if normalization == "salton":
            for col in m.columns:
                for row in m.index:
                    matrix.at[row, col] = m.at[row, col] / np.sqrt(
                        (m.loc[row, row] * m.at[col, col])
                    )
        return matrix

    #
    # MDS
    #
    def mds(x, cmap):

        matplotlib.rc("font", size=10)
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(cmap)

        summary = summary_by_term(
            x,
            column=column,
            top_by=top_by,
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
        )
        terms = summary[column]

        m = matrix.loc[terms, terms]

        # Node sizes prop to Num_Documents
        s = {key: value for key, value in zip(summary[column], summary.Num_Documents)}
        node_sizes = [s[t] for t in terms]
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        node_sizes = [
            600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

        node_colors = [cluster_dict[t] for t in terms]
        cmap = pyplot.cm.get_cmap(cmap)
        node_colors = [cmap(0.2 + 0.75 * i / (n_clusters - 1)) for i in node_colors]

        embedding = MDS(n_components=n_components)
        m_transformed = embedding.fit_transform(m,)
        x0 = [row[0] for row in m_transformed]
        x1 = [row[n_dim] for row in m_transformed]

        ax.scatter(x0, x1, s=node_sizes, linewidths=1, edgecolors="k", c=node_colors)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        dx = 0.1 * (xlim[1] - xlim[0])
        dy = 0.1 * (ylim[1] - ylim[0])

        ax.set_xlim(xlim[0] - dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] + dy)

        for idx, term in enumerate(terms):
            x, y = x0[idx], x1[idx]
            ax.text(
                x
                + 0.01 * (xlim[1] - xlim[0])
                + 0.001 * node_sizes[idx] / 300 * (xlim[1] - xlim[0]),
                y
                - 0.01 * (ylim[1] - ylim[0])
                - 0.001 * node_sizes[idx] / 300 * (ylim[1] - ylim[0]),
                s=term,
                fontsize=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                horizontalalignment="left",
                verticalalignment="top",
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)

        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        return fig

    #
    #
    #  Network map
    #
    #
    def network_map(x, cmap):

        summary = summary_by_term(
            x,
            column=column,
            top_by=top_by,
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
        )
        terms = summary[column]

        m = matrix.loc[terms, terms]

        # Node sizes prop to Num_Documents
        s = {key: value for key, value in zip(summary[column], summary.Num_Documents)}
        node_sizes = [s[t] for t in terms]
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        node_sizes = [
            600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

        # Node colors based on clusters
        node_colors = [cluster_dict[t] for t in terms]
        cmap = pyplot.cm.get_cmap(cmap)
        node_colors = [cmap(0.2 + 0.75 * i / (n_clusters - 1)) for i in node_colors]

        # Draw the network
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()

        G = nx.Graph(ax=ax)
        G.clear()

        # create network nodes
        G.add_nodes_from(terms)

        # create network edges
        n = len(m.columns)

        max_width = 0
        for icol in range(n - 1):
            for irow in range(icol + 1, n):
                link = m.at[m.columns[irow], m.columns[icol]]
                if link > 0:
                    G.add_edge(terms[icol], terms[irow], width=link)
                    if max_width < link:
                        max_width = link

        # Draw edges
        draw_dict = {
            "Circular": nx.draw_circular,
            "Kamada Kawai": nx.draw_kamada_kawai,
            "Planar": nx.draw_planar,
            "Random": nx.draw_random,
            "Spectral": nx.draw_spectral,
            "Spring": nx.draw_spring,
            "Shell": nx.draw_shell,
        }
        draw = draw_dict[layout]

        draw_edges = False
        for e in G.edges.data():
            a, b, width = e
            edge = [(a, b)]
            width = 0.2 + 4.0 * width["width"] / max_width
            draw(
                G,
                ax=ax,
                edgelist=edge,
                width=width,
                edge_color="k",
                with_labels=False,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors="k",
                linewidths=1,
            )
            draw_edges = True

        layout_dict = {
            "Circular": nx.circular_layout,
            "Kamada Kawai": nx.kamada_kawai_layout,
            "Planar": nx.planar_layout,
            "Random": nx.random_layout,
            "Spectral": nx.spectral_layout,
            "Spring": nx.spring_layout,
            "Shell": nx.shell_layout,
        }
        label_pos = layout_dict[layout](G)

        if draw_edges is False:
            nx.draw_networkx_nodes(
                G,
                label_pos,
                ax=ax,
                edge_color="k",
                nodelist=terms,
                node_size=node_sizes,
                node_color=node_colors,
                node_shape="o",
                edgecolors="k",
                linewidths=1,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for idx, term in enumerate(terms):
            x, y = label_pos[term]
            ax.text(
                x
                + 0.01 * (xlim[1] - xlim[0])
                + 0.001 * node_sizes[idx] / 300 * (xlim[1] - xlim[0]),
                y
                - 0.01 * (ylim[1] - ylim[0])
                - 0.001 * node_sizes[idx] / 300 * (ylim[1] - ylim[0]),
                s=term,
                fontsize=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                horizontalalignment="left",
                verticalalignment="top",
            )

        # Figure size
        ax.set_xlim(
            xlim[0] - 0.15 * (xlim[1] - xlim[0]), xlim[1] + 0.15 * (xlim[1] - xlim[0])
        )
        ax.set_ylim(
            ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0])
        )
        ax.set_aspect("equal")
        return fig

    ##
    ## Main
    ##
    matrix = co_occurrence(
        x=x,
        column=column,
        by=column,
        top_by=top_by,
        top_n=top_n,
        limit_to=None,
        exclude=None,
    )

    n_dim = min(n_dim, n_components - 1)

    matrix = normalize(matrix, normalization=normalization)

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(matrix)
    cluster_dict = {
        key: value for key, value in zip(matrix.columns, clustering.labels_)
    }

    if view == "Network":
        return network_map(x=x, cmap=cmap)

    return mds(x=x, cmap=cmap)


#
#
#
#  filepath = "../data/papers/urban-agriculture-CL.csv"
#  df = pd.read_csv(filepath)
# occurrence(
#     x=df,
#     column="Authors",
#     normalization=None,
#     linkage="ward",
#     n_clusters=3,
#     view="network",
#     top_by="Num Documents",
#     top_n=15,
#     limit_to=None,
#     exclude=None,
#     layout="Kamada Kawai",
#     cmap="Greys",
#     figsize=(17, 12),
# )


def occurrence_map(matrix, layout="Kamada Kawai", cmap="Greys", figsize=(17, 12)):
    """Computes the occurrence map directly using networkx.
    """

    if len(matrix.columns) > 50:
        return "Maximum number of nodex exceded!"

    #
    # Data preparation
    #
    terms = matrix.columns.tolist()

    #
    # Node sizes
    #
    node_sizes = [int(w[w.find("[") + 1 : w.find("]")]) for w in terms if "[" in w]
    if len(node_sizes) == 0:
        node_sizes = [500] * len(terms)
    else:
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        if min_size == max_size:
            node_sizes = [500] * len(terms)
        else:
            node_sizes = [
                600 + int(2500 * (w - min_size) / (max_size - min_size))
                for w in node_sizes
            ]

    #
    # Node colors
    #
    cmap = pyplot.cm.get_cmap(cmap)
    node_colors = [
        cmap(0.2 + 0.75 * node_sizes[i] / max(node_sizes))
        for i in range(len(node_sizes))
    ]

    #
    # Remove [...] from text
    #
    terms = [w[: w.find("[")].strip() if "[" in w else w for w in terms]
    matrix.columns = terms
    matrix.index = terms

    #
    # Draw the network
    #
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    draw_dict = {
        "Circular": nx.draw_circular,
        "Kamada Kawai": nx.draw_kamada_kawai,
        "Planar": nx.draw_planar,
        "Random": nx.draw_random,
        "Spectral": nx.draw_spectral,
        "Spring": nx.draw_spring,
        "Shell": nx.draw_shell,
    }
    draw = draw_dict[layout]

    G = nx.Graph(ax=ax)
    G.clear()

    #
    # network nodes
    #
    G.add_nodes_from(terms)

    #
    # network edges
    #
    n = len(matrix.columns)
    max_width = 0
    for icol in range(n - 1):
        for irow in range(icol + 1, n):
            link = matrix.at[matrix.columns[irow], matrix.columns[icol]]
            if link > 0:
                G.add_edge(terms[icol], terms[irow], width=link)
                if max_width < link:
                    max_width = link
    #
    # Draw nodes
    #
    first_time = True
    for e in G.edges.data():
        a, b, width = e
        edge = [(a, b)]
        width = 0.2 + 4.0 * width["width"] / max_width
        draw(
            G,
            ax=ax,
            edgelist=edge,
            width=width,
            edge_color="k",
            with_labels=False,
            font_weight="normal",
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=12,
            horizontalalignment="left",
            verticalalignment="baseline",
            edgecolors="k",
            linewidths=1,
        )
        first_time = False

    #
    # Labels
    #
    layout_dict = {
        "Circular": nx.circular_layout,
        "Kamada Kawai": nx.kamada_kawai_layout,
        "Planar": nx.planar_layout,
        "Random": nx.random_layout,
        "Spectral": nx.spectral_layout,
        "Spring": nx.spring_layout,
        "Shell": nx.shell_layout,
    }
    label_pos = layout_dict[layout](G)

    if not G.edges():
        nx.draw_networkx_nodes(
            G,
            label_pos,
            ax=ax,
            edge_color="k",
            nodelist=terms,
            node_size=node_sizes,
            node_color=node_colors,
            node_shape="o",
            edgecolors="k",
            linewidths=1,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    #
    # Figure size
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for idx, term in enumerate(terms):
        x, y = label_pos[term]
        ax.text(
            x
            + 0.01 * (xlim[1] - xlim[0])
            + 0.001 * node_sizes[idx] / 300 * (xlim[1] - xlim[0]),
            y
            - 0.01 * (ylim[1] - ylim[0])
            - 0.001 * node_sizes[idx] / 300 * (ylim[1] - ylim[0]),
            s=term,
            fontsize=10,
            bbox=dict(
                facecolor="w", alpha=1.0, edgecolor="gray", boxstyle="round,pad=0.5",
            ),
            horizontalalignment="left",
            verticalalignment="top",
        )

    #
    # Figure size
    #

    ax.set_xlim(
        xlim[0] - 0.15 * (xlim[1] - xlim[0]), xlim[1] + 0.15 * (xlim[1] - xlim[0])
    )
    ax.set_ylim(
        ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0])
    )
    ax.set_aspect("equal")
    return fig


def co_occurrence_map(matrix, layout="Kamada Kawai", cmap="Greys", figsize=(17, 12)):
    """Computes the occurrence map directly using networkx.
    """
    #
    def compute_node_sizes(terms):
        #
        node_sizes = [int(w[w.find("[") + 1 : w.find("]")]) for w in terms if "[" in w]
        if len(terms) == 0:
            node_sizes = [500] * len(terms)
        else:
            max_size = max(node_sizes)
            min_size = min(node_sizes)
            if min_size == max_size:
                node_sizes = [500] * len(terms)
            else:
                node_sizes = [
                    600 + int(2500 * (w - min_size) / (max_size - min_size))
                    for w in node_sizes
                ]
        node_colors = [
            cmap(0.2 + 0.75 * node_sizes[i] / max(node_sizes))
            for i in range(len(node_sizes))
        ]
        return node_sizes, node_colors

    #
    #
    #
    cmap = pyplot.cm.get_cmap(cmap)

    #
    # Data preparation
    #
    column_node_sizes, column_node_colors = compute_node_sizes(matrix.columns)
    index_node_sizes, index_node_colors = compute_node_sizes(matrix.index)

    #
    # Remove [...] from text
    #
    terms = matrix.columns.tolist() + matrix.index.tolist()
    terms = [w[: w.find("[")].strip() if "[" in w else w for w in terms]
    matrix.columns = [
        w[: w.find("[")].strip() if "[" in w else w for w in matrix.columns
    ]
    matrix.index = [w[: w.find("[")].strip() if "[" in w else w for w in matrix.index]

    #
    # Draw the network
    #
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    G = nx.Graph(ax=ax)
    G.clear()

    #
    # network nodes
    #
    G.add_nodes_from(terms)

    #
    # network edges
    #
    n = len(matrix.columns)
    max_width = 0
    for col in matrix.columns:
        for row in matrix.index:
            link = matrix.at[row, col]
            if link > 0:
                G.add_edge(row, col, width=link)
                if max_width < link:
                    max_width = link

    draw_dict = {
        "Circular": nx.draw_circular,
        "Kamada Kawai": nx.draw_kamada_kawai,
        "Planar": nx.draw_planar,
        "Random": nx.draw_random,
        "Spectral": nx.draw_spectral,
        "Spring": nx.draw_spring,
        "Shell": nx.draw_shell,
    }
    draw = draw_dict[layout]

    for e in G.edges.data():
        a, b, width = e
        edge = [(a, b)]
        width = 0.2 + 4.0 * width["width"] / max_width
        draw(
            G,
            ax=ax,
            edgelist=edge,
            width=width,
            edge_color="k",
            with_labels=False,
            node_size=1,
        )

    #
    # Layout
    #
    layout_dict = {
        "Circular": nx.circular_layout,
        "Kamada Kawai": nx.kamada_kawai_layout,
        "Planar": nx.planar_layout,
        "Random": nx.random_layout,
        "Spectral": nx.spectral_layout,
        "Spring": nx.spring_layout,
        "Shell": nx.shell_layout,
    }
    pos = layout_dict[layout](G)

    #
    # Draw column nodes
    #
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        edge_color="k",
        nodelist=matrix.columns,
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
        nodelist=matrix.index,
        node_size=index_node_sizes,
        node_color=index_node_colors,
        node_shape="s",
        edgecolors="k",
        linewidths=1,
    )

    #
    # Labels
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    node_sizes = column_node_sizes + index_node_sizes
    for idx, term in enumerate(terms):
        x, y = pos[term]
        ax.text(
            x
            + 0.01 * (xlim[1] - xlim[0])
            + 0.001 * node_sizes[idx] / 300 * (xlim[1] - xlim[0]),
            y
            - 0.01 * (ylim[1] - ylim[0])
            - 0.001 * node_sizes[idx] / 300 * (ylim[1] - ylim[0]),
            s=term,
            fontsize=10,
            bbox=dict(
                facecolor="w", alpha=1.0, edgecolor="gray", boxstyle="round,pad=0.5",
            ),
            horizontalalignment="left",
            verticalalignment="top",
        )

    #
    # Figure size
    #
    ax.set_xlim(
        xlim[0] - 0.15 * (xlim[1] - xlim[0]), xlim[1] + 0.15 * (xlim[1] - xlim[0])
    )
    ax.set_ylim(
        ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0])
    )
    ax.set_aspect("equal")
    return fig


def slope_chart(matrix, x, column, by, top_by, figsize, cmap="Blues"):
    """
    """
    matplotlib.rc("font", size=12)

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = pyplot.cm.get_cmap(cmap)

    matrix.columns = [a[: a.find("[")].strip() for a in matrix.columns]
    matrix.index = [a[: a.find("[")].strip() for a in matrix.index]

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
    # left y-axis
    #

    # sizes
    z = summary_by_term(x, column=by, top_by=top_by, top_n=None)
    dic = {key: value for key, value in zip(z[z.columns[0]], z["Num_Documents"])}
    sizes = [dic[index] for index in matrix.index]
    sizes = [80 + 300 * (w - min(sizes)) / (max(sizes) - min(sizes)) for w in sizes]

    #  color
    dic = {key: value for key, value in zip(z[z.columns[0]], z["Times_Cited"])}
    colors = [dic[index] for index in matrix.index]
    colors = [
        cmap((0.25 + 0.75 * (value - min(colors)) / (max(colors) - min(colors))))
        for value in colors
    ]

    ax.scatter(
        x=[1] * m, y=yleft, s=sizes, c=colors, zorder=10, linewidths=1, edgecolors="k"
    )

    for idx, text in enumerate(matrix.index):
        ax.text(0.9, yleft[idx], text, ha="right", va="center")

    #
    # right y-axis
    #

    #  sizes
    z = summary_by_term(x, column=column, top_by=top_by, top_n=None)
    dic = {key: value for key, value in zip(z[z.columns[0]], z["Num_Documents"])}
    sizes = [dic[column] for column in matrix.columns]
    sizes = [80 + 300 * (w - min(sizes)) / (max(sizes) - min(sizes)) for w in sizes]

    #  color
    dic = {key: value for key, value in zip(z[z.columns[0]], z["Times_Cited"])}
    colors = [dic[col] for col in matrix.columns]
    colors = [
        cmap((0.25 + 0.75 * (value - min(colors)) / (max(colors) - min(colors))))
        for value in colors
    ]

    ax.scatter(
        x=[3] * n, y=yright, s=sizes, c=colors, zorder=10, linewidths=1, edgecolors="k"
    )

    for idx, text in enumerate(matrix.columns):
        ax.text(3.1, yright[idx], text, ha="left", va="center")

    ax.invert_yaxis()
    ax.axis("off")

    return fig


# import matplotlib
# import numpy as np

# filepath = "../data/papers/urban-agriculture-CL.csv"
# df = pd.read_csv(filepath)
# matrix = co_occurrence(
#     x=df,
#     column="Authors",
#     by="Source_title",
#     top_by="Num Documents",
#     top_n=10,
#     limit_to=None,
#     exclude=None,
# )
# slope_chart(
#     matrix,
#     x=df,
#     column="Authors",
#     by="Source_title",
#     top_by=None,
#     figsize=(6, 6),
#     cmap="Blues",
# )


def correspondence_matrix(
    x,
    column_IDX,
    column_COL,
    sep_IDX=None,
    sep_COL=None,
    as_matrix=False,
    keywords=None,
):
    """

    """
    result = co_occurrence(
        x,
        column_IDX=column_IDX,
        column_COL=column_COL,
        sep_IDX=sep_IDX,
        sep_COL=sep_COL,
        as_matrix=True,
        minmax=None,
        keywords=keywords,
    )

    matrix = result.transpose().values
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

    return pd.DataFrame(data=rowScores, columns=result.columns, index=result.columns)


# def relationship(x, y):
#     sxy = sum([a * b * min(a, b) for a, b in zip(x, y)])
#     a = math.sqrt(sum(x))
#     b = math.sqrt(sum(y))
#     return sxy / (a * b)


###############################################################################
##
##  APP
##
###############################################################################

WIDGET_WIDTH = "180px"
LEFT_PANEL_HEIGHT = "710px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "770px", 0]

#
#
#  Co-occurrence Matrix
#
#


def __TAB0__(x, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
    controls = [
        # 0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Matrix", "Heatmap", "Bubble plot", "Network", "Slope chart"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "by",
            "desc": "By Column:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Times Cited"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(5, 51, 5)), layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 6
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=[
                    "Alphabetic asc.",
                    "Alphabetic desc.",
                    "Frequency/Cited by asc.",
                    "Frequency/Cited by desc.",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 7
        {
            "arg": "layout",
            "desc": "Map layout:",
            "widget": widgets.Dropdown(
                options=[
                    "Circular",
                    "Kamada Kawai",
                    "Planar",
                    "Random",
                    "Spectral",
                    "Spring",
                    "Shell",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 8
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 9
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
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
        by = kwargs["by"]
        view = kwargs["view"]
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        sort_by = kwargs["sort_by"]
        layout = kwargs["layout"]
        figsize_width = int(kwargs["figsize_width"])
        figsize_height = int(kwargs["figsize_height"])
        #
        matrix = co_occurrence(
            x,
            column=column,
            by=by,
            top_by=top_by,
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
        )
        #
        if view == "Matrix":
            controls[-2]["widget"].disabled = True
            controls[-1]["widget"].disabled = True
        else:
            controls[-2]["widget"].disabled = False
            controls[-1]["widget"].disabled = False
        if view == "Network":
            controls[7]["widget"].disabled = False
        else:
            controls[7]["widget"].disabled = True
        #
        if top_by == "Num Documents":
            s = summary_by_term(
                x=x,
                column=column,
                top_by=top_by,
                top_n=top_n,
                limit_to=limit_to,
                exclude=exclude,
            )
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[column].tolist(), s["Num_Documents"].tolist())
            }
            matrix = matrix.rename(columns=new_names)
            #
            if column == by:
                matrix = matrix.rename(index=new_names)
            else:
                s = summary_by_term(x, by)
                new_names = {
                    a: "{} [{:d}]".format(a, b)
                    for a, b in zip(s[by].tolist(), s["Num_Documents"].tolist())
                }
                matrix = matrix.rename(index=new_names)
        else:
            s = summary_by_term(
                x=x,
                column=column,
                top_by=top_by,
                top_n=top_n,
                limit_to=limit_to,
                exclude=exclude,
            )
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[column].tolist(), s["Times_Cited"].tolist())
            }
            matrix = matrix.rename(columns=new_names)
            #
            if column == by:
                matrix = matrix.rename(index=new_names)
            else:
                s = summary_by_term(x, by)
                new_names = {
                    a: "{} [{:d}]".format(a, b)
                    for a, b in zip(s[by].tolist(), s["Times_Cited"].tolist())
                }
                matrix = matrix.rename(index=new_names)

        #
        # Sort order
        #
        g = (
            lambda m: m[m.find("[") + 1 : m.find("]")].zfill(5)
            + " "
            + m[: m.find("[") - 1]
        )
        if sort_by == "Frequency/Cited by asc.":
            col_names = sorted(matrix.columns, key=g, reverse=False)
            row_names = sorted(matrix.index, key=g, reverse=False)
            matrix = matrix.loc[row_names, col_names]
        if sort_by == "Frequency/Cited by desc.":
            col_names = sorted(matrix.columns, key=g, reverse=True)
            row_names = sorted(matrix.index, key=g, reverse=True)
            matrix = matrix.loc[row_names, col_names]
        if sort_by == "Alphabetic asc.":
            matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
                axis=1, ascending=True
            )
        if sort_by == "Alphabetic desc.":
            matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
                axis=1, ascending=False
            )
        #
        output.clear_output()
        with output:
            #
            # View
            #
            if view == "Matrix":
                with pd.option_context(
                    "display.max_columns", 50, "display.max_rows", 50
                ):
                    display(matrix.style.background_gradient(cmap=cmap, axis=None))
            if view == "Heatmap":
                display(
                    plt.heatmap(
                        matrix, cmap=cmap, figsize=(figsize_width, figsize_height)
                    )
                )
            if view == "Bubble plot":
                display(
                    plt.bubble(
                        matrix.transpose(),
                        axis=0,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if view == "Network" and column == by:
                display(
                    occurrence_map(
                        matrix,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if view == "Network" and column != by:
                display(
                    co_occurrence_map(
                        matrix,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )

            if view == "Slope chart":
                display(
                    slope_chart(
                        matrix=matrix,
                        x=x,
                        column=column,
                        by=by,
                        top_by=top_by,
                        figsize=(figsize_width, figsize_height),
                        cmap=cmap,
                    )
                )

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                    if control["desc"] not in ["Figsize"]
                ]
                + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([controls[-2]["widget"], controls[-1]["widget"],]),
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


def __TAB1__(x, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
    controls = [
        # 0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Network", "MDS"], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Times Cited"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(5, 81, 5)), layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "normalization",
            "desc": "Normalization:",
            "widget": widgets.Dropdown(
                options=["None", "association", "inclusion", "jaccard", "salton"],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 6
        {
            "arg": "n_clusters",
            "desc": "Num clusters:",
            "widget": widgets.Dropdown(
                options=list(range(3, 21)), layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 7
        {
            "arg": "n_components",
            "desc": "Num components:",
            "widget": widgets.Dropdown(
                options=list(range(2, 10)), layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 8
        {
            "arg": "n_dim",
            "desc": "Dim for plotting:",
            "widget": widgets.Dropdown(
                options=list(range(1, 10)), layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 9
        {
            "arg": "layout",
            "desc": "Map layout:",
            "widget": widgets.Dropdown(
                options=[
                    "Circular",
                    "Kamada Kawai",
                    "Planar",
                    "Random",
                    "Spectral",
                    "Spring",
                    "Shell",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 10
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 11
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
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
        view = kwargs["view"]
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        n_clusters = int(kwargs["n_clusters"])
        n_components = int(kwargs["n_components"])
        n_dim = int(kwargs["n_dim"])
        normalization = kwargs["normalization"]
        layout = kwargs["layout"]
        figsize_width = int(kwargs["figsize_width"])
        figsize_height = int(kwargs["figsize_height"])
        #
        if view == "Network":
            controls[7]["widget"].disabled = True
            controls[8]["widget"].disabled = True
            controls[9]["widget"].disabled = False
        if view == "MDS":
            controls[7]["widget"].disabled = False
            controls[8]["widget"].disabled = False
            controls[9]["widget"].disabled = True

        #
        output.clear_output()
        with output:
            #
            # View
            #
            if view == "Network":
                display(
                    occurrence(
                        x,
                        column=column,
                        normalization=normalization,
                        linkage="ward",
                        n_clusters=n_clusters,
                        view=view,
                        top_by=top_by,
                        top_n=top_n,
                        limit_to=None,
                        exclude=None,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
                return
            if view == "MDS":
                display(
                    occurrence(
                        x,
                        column=column,
                        normalization=normalization,
                        linkage="ward",
                        n_clusters=n_clusters,
                        n_components=n_components,
                        n_dim=n_dim,
                        view=view,
                        top_by=top_by,
                        top_n=top_n,
                        limit_to=None,
                        exclude=None,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
                return

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                    if control["desc"] not in ["Figsize"]
                ]
                + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([controls[-2]["widget"], controls[-1]["widget"],]),
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


#
#
#  Occurrence Network
#
#


def app(df, limit_to=None, exclude=None):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__TAB0__(df, limit_to, exclude), __TAB1__(df, limit_to, exclude)]
    body.set_title(0, "Co-occurrence")
    body.set_title(1, "Occurrence")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Co-occurrence Analysis"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )
