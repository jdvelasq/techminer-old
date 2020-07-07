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
import techminer.common as common
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


def corr(
    data,
    column,
    by=None,
    method="pearson",
    output=0,
    top_by=None,
    top_n=None,
    sort_by=None,
    ascending=True,
    cmap=None,
    layout="Circular",
    min_link_value=-1,
    figsize=(8, 8),
    limit_to=None,
    exclude=None,
):
    """Computes cross-correlation among items in two different columns of the dataframe.

    Args:
        column_IDX (str): the first column.
        sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
        column_COL (str): the second column.
        sep_COL (str): Character used as internal separator for the elements in the column_COL.
        method (str): Available methods are:

            - pearson : Standard correlation coefficient.

            - kendall : Kendall Tau correlation coefficient.

            - spearman : Spearman rank correlation.

        as_matrix (bool): the result is reshaped by melt or not.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
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
    5     A;B               d            5   5


    >>> document_term_matrix(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> document_term_matrix(df, 'Author_Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1
    5  0  0  0  1


    >>> corr(df, 'Authors', 'Author_Keywords')
              A         B         C        D
    A  1.000000 -1.000000 -0.333333 -0.57735
    B -1.000000  1.000000  0.333333  0.57735
    C -0.333333  0.333333  1.000000  0.57735
    D -0.577350  0.577350  0.577350  1.00000

    >>> corr(df, 'Authors', 'Author_Keywords', limit_to=['A', 'B', 'C'])
              A         B         C
    A  1.000000 -1.000000 -0.333333
    B -1.000000  1.000000  0.333333
    C -0.333333  0.333333  1.000000

    >>> corr(df, 'Authors', 'Author_Keywords', exclude=['A'])
              B         C        D
    B  1.000000  0.333333  0.57735
    C  0.333333  1.000000  0.57735
    D  0.577350  0.577350  1.00000

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
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
    5     A;B               d            5   5

    >>> document_term_matrix(df, column='Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> corr(df, 'Authors')
              A         B         C         D
    A  1.000000 -0.316228  0.316228 -0.632456
    B -0.316228  1.000000  0.200000  0.200000
    C  0.316228  0.200000  1.000000 -0.200000
    D -0.632456  0.200000 -0.200000  1.000000


    >>> corr(df, 'Author_Keywords')
          a     b     c     d
    a  1.00  0.25 -0.50 -0.50
    b  0.25  1.00 -0.50 -0.50
    c -0.50 -0.50  1.00  0.25
    d -0.50 -0.50  0.25  1.00



    """
    x = data.copy()

    if by is None:
        by = column

    if column == by:

        dtm = document_term_matrix(x, column=column)
        dtm = filter_index(
            x=x,
            column=column,
            matrix=dtm,
            axis=1,
            top_by=0,
            top_n=top_n,
            sort_by=sort_by,
            ascending=ascending,
            limit_to=limit_to,
            exclude=exclude,
        )
        result = dtm.corr(method=method)

    else:

        w = x[[column, by, "ID"]].dropna()
        A = document_term_matrix(w, column=column)
        A = filter_index(
            x=x,
            column=column,
            matrix=A,
            axis=1,
            top_by=0,
            top_n=top_n,
            sort_by=sort_by,
            ascending=ascending,
            limit_to=limit_to,
            exclude=exclude,
        )
        B = document_term_matrix(w, column=by)
        matrix = np.matmul(B.transpose().values, A.values)
        matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)
        result = matrix.corr(method=method)

    if isinstance(output, str):
        output = {"Matrix": 0, "Heatmap": 1, "Correlation map": 2, "Chord diagram": 3,}[
            output
        ]

    if output == 0:

        summ = by_term.analytics(
            data=x,
            column=column,
            output=0,
            top_by=0,
            top_n=None,
            sort_by=0,
            ascending=ascending,
            limit_to=limit_to,
            exclude=exclude,
        )
        fmt = _get_fmt(summ)
        new_names = {
            key: fmt.format(key, nd, tc)
            for key, nd, tc in zip(summ.index, summ.Num_Documents, summ.Times_Cited)
        }
        result.columns = [new_names[w] for w in result.columns]
        result.index = [new_names[w] for w in result.index]

        if cmap is None:
            return result
        else:
            return result.style.format(
                lambda q: "{:+4.3f}".format(q) if q >= min_link_value else ""
            ).background_gradient(cmap=cmap)

    if output == 1:
        return plt.heatmap(result, cmap=cmap, figsize=figsize)

    if output == 2:
        return correlation_map(
            matrix=result,
            summary=by_term.analytics(x, column=column, top_by=None, top_n=None),
            layout=layout,
            cmap=cmap,
            figsize=figsize,
            min_link_value=min_link_value,
        )

    if output == 3:
        return correlation_chord_diagram(
            matrix=result,
            summary=by_term.analytics(x, column=column, top_by=None, top_n=None),
            cmap=cmap,
            figsize=figsize,
            min_link_value=min_link_value,
        )

    return result


def _get_fmt(summ):
    n_Num_Documents = int(np.log10(summ["Num_Documents"].max())) + 1
    n_Times_Cited = int(np.log10(summ["Times_Cited"].max())) + 1
    return "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"


def correlation_chord_diagram(
    matrix, summary, cmap="Greys", figsize=(17, 12), min_link_value=0,
):
    x = matrix.copy()
    # ---------------------------------------------------
    #
    # Node sizes
    #
    #
    # Data preparation
    #
    terms = matrix.columns.tolist()
    terms = [w[: w.find("[")].strip() if "[" in w else w for w in terms]
    terms = [w.strip() for w in terms]

    num_documents = {k: v for k, v in zip(summary.index, summary["Num_Documents"])}
    times_cited = {k: v for k, v in zip(summary.index, summary["Times_Cited"])}

    #
    # Node sizes
    #
    node_sizes = [num_documents[t] for t in terms]
    min_size = min(node_sizes)
    max_size = max(node_sizes)
    if min_size == max_size:
        node_sizes = [500 for t in terms]
    else:
        node_sizes = [
            600 + int(2500 * (t - min_size) / (max_size - min_size)) for t in node_sizes
        ]

    #
    # Node colors
    #
    cmap = pyplot.cm.get_cmap(cmap)
    node_colors = [times_cited[t] for t in terms]
    min_color = min(node_colors)
    max_color = max(node_colors)
    if min_color == max_color:
        node_colors = [cmap(0.8) for t in terms]
    else:
        node_colors = [
            cmap(0.2 + 0.75 * (t - min_color) / (max_color - min_color))
            for t in node_colors
        ]

    #
    # ---------------------------------------------------

    cd = ChordDiagram()
    for idx, term in enumerate(x.columns):
        cd.add_node(term, color=node_colors[idx], s=node_sizes[idx])

    style = ["-", "-", "--", ":"]
    #  if solid_lines is True:
    #      style = list("----")

    width = [4, 2, 1, 1]
    #  if solid_lines is True:
    #      width = [4, 2, 1, 1]

    links = 0
    for idx_col in range(len(x.columns) - 1):
        for idx_row in range(idx_col + 1, len(x.columns)):

            node_a = x.index[idx_row]
            node_b = x.columns[idx_col]
            value = x[node_b][node_a]

            if value > 0.75 and value >= min_link_value:
                cd.add_edge(
                    node_a,
                    node_b,
                    linestyle=style[0],
                    linewidth=width[0],
                    color="black",
                )
                links += 1
            elif value > 0.50 and value >= min_link_value:
                cd.add_edge(
                    node_a,
                    node_b,
                    linestyle=style[1],
                    linewidth=width[1],
                    color="black",
                )
                links += 1
            elif value > 0.25 and value >= min_link_value:
                cd.add_edge(
                    node_a,
                    node_b,
                    linestyle=style[2],
                    linewidth=width[2],
                    color="black",
                )
                links += 1
            elif value <= 0.25 and value >= min_link_value and value > 0.0:
                cd.add_edge(
                    node_a,
                    node_b,
                    linestyle=style[3],
                    linewidth=width[3],
                    color="black",
                )
                links += 1

    return cd.plot(figsize=figsize)


def correlation_map(
    matrix,
    summary,
    layout="Kamada Kawai",
    cmap="Greys",
    figsize=(17, 12),
    min_link_value=0,
):
    """Computes the correlation map directly using networkx.
    """

    if len(matrix.columns) > 50:
        return "Maximum number of nodex exceded!"

    #
    # Networkx
    #
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    #
    # Data preparation
    #
    terms = matrix.columns.tolist()
    terms = [w[: w.find("[")].strip() if "[" in w else w for w in terms]
    terms = [w.strip() for w in terms]

    num_documents = {k: v for k, v in zip(summary.index, summary["Num_Documents"])}
    times_cited = {k: v for k, v in zip(summary.index, summary["Times_Cited"])}

    #
    # Node sizes
    #
    node_sizes = [num_documents[t] for t in terms]
    min_size = min(node_sizes)
    max_size = max(node_sizes)
    if min_size == max_size:
        node_sizes = [500 for t in terms]
    else:
        node_sizes = [
            600 + int(2500 * (t - min_size) / (max_size - min_size)) for t in node_sizes
        ]

    #
    # Node colors
    #
    cmap = pyplot.cm.get_cmap(cmap)
    node_colors = [times_cited[t] for t in terms]
    min_color = min(node_colors)
    max_color = max(node_colors)
    if min_color == max_color:
        node_colors = [cmap(0.8) for t in terms]
    else:
        node_colors = [
            cmap(0.2 + 0.75 * (t - min_color) / (max_color - min_color))
            for t in node_colors
        ]
    matrix.columns = terms
    matrix.index = terms

    #
    # Add nodes
    #
    G.add_nodes_from(terms)

    #
    # Draw the network
    #
    n = len(matrix.columns)
    edges = []
    n_edges_75 = 0
    n_edges_50 = 0
    n_edges_25 = 0
    n_edges_0 = 0

    for icol in range(n):
        for irow in range(icol + 1, n):
            if (
                min_link_value is None
                or matrix[terms[icol]][terms[irow]] >= min_link_value
            ):
                link = None
                if matrix[terms[icol]][terms[irow]] > 0.75:
                    link = 1
                    n_edges_75 += 1
                elif matrix[terms[icol]][terms[irow]] > 0.50:
                    link = 2
                    n_edges_50 += 1
                elif matrix[terms[icol]][terms[irow]] > 0.25:
                    link = 3
                    n_edges_25 += 1
                elif matrix[terms[icol]][terms[irow]] > 0.00:
                    link = 4
                    n_edges_0 += 1
                if link is not None:
                    G.add_edge(terms[icol], terms[irow], link=link)

    pos = {
        "Circular": nx.circular_layout,
        "Kamada Kawai": nx.kamada_kawai_layout,
        "Planar": nx.planar_layout,
        "Random": nx.random_layout,
        "Spectral": nx.spectral_layout,
        "Spring": nx.spring_layout,
        "Shell": nx.shell_layout,
    }[layout](G)

    #
    # Network drawing
    #
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
        a, b, link = e
        edge = [(a, b)]
        link = link["link"]
        #  width = 4
        #  style = "solid"
        #  edge_color = "k"
        if link == 1:
            width = 4
            style = "solid"
            edge_color = "k"
        if link == 2:
            width = 2
            style = "solid"
            edge_color = "k"
        if link == 3:
            width = 1
            style = "dashed"
            edge_color = "k"
        if link == 4:
            width = 1
            style = "dotted"
            edge_color = "k"

        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=edge,
            width=width,
            edge_color=edge_color,
            style=style,
            with_labels=False,
            node_size=1,
        )

    #
    # Draw column nodes
    #

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        edge_color="k",
        nodelist=matrix.columns.tolist(),
        node_size=node_sizes,
        node_color=node_colors,
        node_shape="o",
        edgecolors="k",
        linewidths=1,
    )

    #
    # Figure size
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    common.ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)

    #
    # Figure size
    #
    common.ax_expand_limits(ax)

    #
    # Legend
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_len = (xlim[1] - xlim[0]) / 50
    y_len = (ylim[1] - ylim[0]) / 50
    text_len = max(x_len, y_len)
    #
    text_75 = "> 0.75 ({})".format(n_edges_75)
    text_50 = "0.50-0.75 ({})".format(n_edges_50)
    text_25 = "0.25-0.50 ({})".format(n_edges_25)
    text_0 = "< 0.25 ({})".format(n_edges_0)
    #

    #
    ax.text(xlim[0] + 2.5 * text_len, ylim[0] + text_len * 3, text_75)
    ax.text(xlim[0] + 2.5 * text_len, ylim[0] + text_len * 2, text_50)
    ax.text(xlim[0] + 2.5 * text_len, ylim[0] + text_len * 1, text_25)
    ax.text(xlim[0] + 2.5 * text_len, ylim[0] + text_len * 0, text_0)
    #
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * text_len],
        [ylim[0] + text_len * 0.25, ylim[0] + text_len * 0.25],
        "k:",
        linewidth=1,
    )
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * text_len],
        [ylim[0] + text_len * 1.25, ylim[0] + text_len * 1.25],
        "k--",
        linewidth=1,
    )
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * text_len],
        [ylim[0] + text_len * 2.25, ylim[0] + text_len * 2.25],
        "k-",
        linewidth=2,
    )
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * text_len],
        [ylim[0] + text_len * 3.25, ylim[0] + text_len * 3.25],
        "k-",
        linewidth=4,
    )

    ax.axis("off")

    return fig


###############################################################################
##
##  APP
##
###############################################################################


def __TAB0__(x, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        gui.dropdown(
            desc="View:",
            options=["Matrix", "Heatmap", "Correlation map", "Chord diagram"],
        ),
        gui.dropdown(desc="Term:", options=[z for z in COLUMNS if z in x.columns],),
        gui.dropdown(desc="By:", options=[z for z in COLUMNS if z in x.columns],),
        gui.dropdown(desc="Method:", options=["pearson", "kendall", "spearman"],),
        gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
        gui.top_n(),
        gui.dropdown(
            desc="Sort by:", options=["Alphabetic", "Num Documents", "Times Cited",],
        ),
        gui.ascending(),
        gui.dropdown(desc="Min link value:", options="0.00 0.25 0.50 0.75".split(" "),),
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
        column = kwargs["term"]
        by = kwargs["by"]
        method = kwargs["method"]
        min_link_value = float(kwargs["min_link_value"].split(" ")[0])
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        view = kwargs["view"]
        sort_by = kwargs["sort_by"]
        ascending = kwargs["ascending"]
        layout = kwargs["layout"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        #
        if view == "Matrix" or view == "Heatmap":
            left_panel[-7]["widget"].disabled = False
            left_panel[-6]["widget"].disabled = False
            left_panel[-5]["widget"].disabled = False
            left_panel[-3]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = True
            left_panel[-1]["widget"].disabled = True
        if view == "Heatmap":
            left_panel[-7]["widget"].disabled = False
            left_panel[-6]["widget"].disabled = False
            left_panel[-5]["widget"].disabled = False
            left_panel[-3]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = False
            left_panel[-1]["widget"].disabled = False
        if view == "Correlation map":
            left_panel[-7]["widget"].disabled = True
            left_panel[-6]["widget"].disabled = True
            left_panel[-5]["widget"].disabled = True
            left_panel[-3]["widget"].disabled = False
            left_panel[-2]["widget"].disabled = False
            left_panel[-1]["widget"].disabled = False
        if view == "Chord diagram":
            left_panel[-7]["widget"].disabled = True
            left_panel[-6]["widget"].disabled = True
            left_panel[-5]["widget"].disabled = True
            left_panel[-3]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = False
            left_panel[-1]["widget"].disabled = False
        #

        #
        output.clear_output()
        with output:
            display(
                corr(
                    x,
                    column=column,
                    by=by,
                    method=method,
                    output=view,
                    cmap=cmap,
                    top_by=top_by,
                    top_n=top_n,
                    sort_by=sort_by,
                    ascending=ascending,
                    layout=layout,
                    limit_to=limit_to,
                    exclude=exclude,
                    figsize=(width, height),
                )
            )

        return
        #
        #
        #
        # if top_by == "Num Documents":
        #     s = summary_by_term(x, column)
        #     new_names = {
        #         a: "{} [{:d}]".format(a, b)
        #         for a, b in zip(s[column].tolist(), s["Num_Documents"].tolist())
        #     }
        # else:
        #     s = summary_by_term(x, column)
        #     new_names = {
        #         a: "{} [{:d}]".format(a, b)
        #         for a, b in zip(s[column].tolist(), s["Times_Cited"].tolist())
        #     }
        # matrix = matrix.rename(columns=new_names, index=new_names)

        # output.clear_output()
        # with output:
        #     if view == "Matrix" or view == "Heatmap":
        #         #
        #         # Sort order
        #         #
        #         g = (
        #             lambda m: m[m.find("[") + 1 : m.find("]")].zfill(5)
        #             + " "
        #             + m[: m.find("[") - 1]
        #         )
        #         if sort_by == "Frequency/Cited by asc.":
        #             names = sorted(matrix.columns, key=g, reverse=False)
        #             matrix = matrix.loc[names, names]
        #         if sort_by == "Frequency/Cited by desc.":
        #             names = sorted(matrix.columns, key=g, reverse=True)
        #             matrix = matrix.loc[names, names]
        #         if sort_by == "Alphabetic asc.":
        #             matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
        #                 axis=1, ascending=True
        #             )
        #         if sort_by == "Alphabetic desc.":
        #             matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
        #                 axis=1, ascending=False
        #             )
        #         #
        #         # View
        #         #
        #         with pd.option_context(
        #             "display.max_columns", 60, "display.max_rows", 60
        #         ):
        #             if view == "Matrix":
        #                 display(
        #                     matrix.style.format(
        #                         lambda q: "{:+4.3f}".format(q)
        #                         if q >= min_link_value
        #                         else ""
        #                     ).background_gradient(cmap=cmap)
        #                 )
        #         if view == "Heatmap":
        #             display(
        #                 plt.heatmap(
        #                     matrix, cmap=cmap, figsize=(figsize_width, figsize_height)
        #                 )
        #             )
        #         #
        #     if view == "Correlation map":
        #         #
        #         display(
        #             correlation_map(
        #                 matrix=matrix,
        #                 summary=summary_by_term(
        #                     x, column=column, top_by=None, top_n=None
        #                 ),
        #                 layout=layout,
        #                 cmap=cmap,
        #                 figsize=(figsize_width, figsize_height),
        #                 min_link_value=min_link_value,
        #             )
        #         )
        #         #
        #     if view == "Chord diagram":
        #         #
        #         display(
        #             chord_diagram(
        #                 matrix,
        #                 figsize=(figsize_width, figsize_height),
        #                 cmap=cmap,
        #                 minval=min_link_value,
        #             )
        #         )

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))

    grid = GridspecLayout(13, 6)
    #
    # Left panel
    #
    for index in range(len(left_panel)):
        grid[index, 0] = widgets.HBox(
            [
                widgets.Label(value=left_panel[index]["desc"]),
                left_panel[index]["widget"],
            ],
            layout=Layout(
                display="flex", justify_content="flex-end", align_content="center",
            ),
        )
    #
    # Output
    #
    grid[0:, 1:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )

    return grid


#
#
# APP
#
#
def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Correlation",
        tab_titles=["Correlation Analysis",],
        tab_widgets=[__TAB0__(data, limit_to=limit_to, exclude=exclude),],
        tab=tab,
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
