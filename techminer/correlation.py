"""
Correlation Analysis
==================================================================================================


"""

import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout

#  from techminer.by_term import citations_by_term, documents_by_term, summary_by_term
from techminer.by_term import summary_by_term
from techminer.co_occurrence import (
    co_occurrence,
    compute_tfm,
    #    most_cited_by,
    #    most_frequent,
)
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.keywords import Keywords
from techminer.maps import Map
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS, chord_diagram


def corr(
    x,
    column,
    by=None,
    method="pearson",
    top_by=None,
    top_n=None,
    cmap=None,
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
    ...       'Cited_by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author_Keywords  Cited_by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4
    5     A;B               d         5   5


    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> compute_tfm(df, 'Author_Keywords')
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
    ...       'Cited_by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author_Keywords  Cited_by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4
    5     A;B               d         5   5

    >>> compute_tfm(df, column='Authors')
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
    if by is None:
        by = column
        # if isinstance(limit_to, dict) and column in limit_to.keys():
        #     limit_to[by] = limit_to[column]
        # if isinstance(exclude, dict) and column in exclude.keys():
        #     exclude[by] = exclude[column]

    if column == by:
        #
        tfm = compute_tfm(x, column=column, limit_to=limit_to, exclude=exclude)
        #
        if limit_to is None or column not in limit_to.keys():
            tfm = tfm[most_frequent(x, column, top_n, limit_to=None, exclude=exclude)]
        result = tfm.corr(method=method)
    else:
        w = x[[column, by, "ID"]].dropna()
        #
        A = compute_tfm(w, column, limit_to, exclude)
        #
        if limit_to is None or column not in limit_to.keys():
            A = A[most_frequent(x, column, top_n, limit_to=None, exclude=exclude)]
        B = compute_tfm(w, by)
        tfm = np.matmul(B.transpose().values, A.values)
        tfm = pd.DataFrame(tfm, columns=A.columns, index=B.columns)
        result = tfm.corr(method=method)

    result = result.sort_index(axis=0, ascending=True)
    result = result.sort_index(axis=1, ascending=True)
    return result


def correlation_map(
    matrix, layout="Kamada Kawai", cmap="Greys", figsize=(17, 12), min_link_value=0
):
    """Computes the correlation map directly using networkx.
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
    n = len(matrix.columns)
    edges_75 = []
    edges_50 = []
    edges_25 = []
    other_edges = []

    for icol in range(n):
        for irow in range(icol + 1, n):
            if (
                min_link_value is None
                or matrix[terms[icol]][terms[irow]] >= min_link_value
            ):
                if matrix[terms[icol]][terms[irow]] > 0.75:
                    edges_75.append((terms[icol], terms[irow]))
                elif matrix[terms[icol]][terms[irow]] > 0.50:
                    edges_50.append((terms[icol], terms[irow]))
                elif matrix[terms[icol]][terms[irow]] > 0.25:
                    edges_25.append((terms[icol], terms[irow]))
                elif matrix[terms[icol]][terms[irow]] > 0.0:
                    other_edges.append((terms[icol], terms[irow]))

    if len(edges_75) == 0:
        edges_75 = None
    if len(edges_50) == 0:
        edges_50 = None
    if len(edges_25) == 0:
        edges_25 = None
    if len(other_edges) == 0:
        other_edges = None

    #
    # Network drawing
    #
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
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

    G = nx.Graph(ax=ax)
    G.clear()
    #
    G.add_nodes_from(terms)
    #
    if edges_75 is not None:
        G.add_edges_from(edges_75)
    if edges_50 is not None:
        G.add_edges_from(edges_50)
    if edges_25 is not None:
        G.add_edges_from(edges_25)
    if other_edges is not None:
        G.add_edges_from(other_edges)
    #
    with_labels = True
    if edges_75 is not None:
        draw(
            G,
            ax=ax,
            edgelist=edges_75,
            width=4,
            edge_color="k",
            with_labels=False,
            font_weight="bold",
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=10,
            horizontalalignment="left",
            verticalalignment="baseline",
            edgecolors="k",
            linewidths=2,
        )
        with_labels = False
    #
    if edges_50 is not None:
        draw(
            G,
            ax=ax,
            edgelist=edges_50,
            edge_color="k",
            width=2,
            with_labels=False,
            font_weight=with_labels,
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=10,
            horizontalalignment="left",
            verticalalignment="baseline",
            edgecolors="k",
            linewidths=2,
        )
        with_labels = False
    #
    if edges_25 is not None:
        draw(
            G,
            ax=ax,
            edgelist=edges_25,
            edge_color="k",
            width=1,
            style="dashed",
            alpha=1.0,
            with_labels=False,
            font_weight=with_labels,
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=10,
            horizontalalignment="left",
            verticalalignment="baseline",
            edgecolors="k",
            linewidths=2,
        )
        with_labels = False
    if other_edges is not None:
        draw(
            G,
            ax=ax,
            edgelist=other_edges,
            edge_color="k",
            width=1,
            alpha=1.0,
            style="dotted",
            with_labels=False,
            font_weight=with_labels,
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=10,
            horizontalalignment="left",
            verticalalignment="baseline",
            edgecolors="k",
            linewidths=2,
        )
        with_labels = False

    if with_labels is True:
        draw(
            G,
            ax=ax,
            edgelist=None,
            edge_color="k",
            width=1,
            alpha=1.0,
            style="dotted",
            with_labels=False,
            font_weight=with_labels,
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=10,
            horizontalalignment="left",
            verticalalignment="baseline",
            edgecolors="k",
            linewidths=2,
        )

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
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(
        xlim[0] - 0.15 * (xlim[1] - xlim[0]), xlim[1] + 0.15 * (xlim[1] - xlim[0])
    )
    ax.set_ylim(
        ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0])
    )
    #
    # Legend
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_len = (xlim[1] - xlim[0]) / 40
    y_len = (ylim[1] - ylim[0]) / 40
    #
    text_75 = "> 0.75 ({})".format(len(edges_75) if edges_75 is not None else 0)
    text_50 = "0.50-0.75 ({})".format(len(edges_50) if edges_50 is not None else 0)
    text_25 = "0.25-0.50 ({})".format(len(edges_25) if edges_25 is not None else 0)
    text_0 = "< 0.25 ({})".format(len(other_edges) if other_edges is not None else 0)
    #
    ax.text(xlim[0] + 2.5 * x_len, ylim[0] + y_len * 3, text_75)
    ax.text(xlim[0] + 2.5 * x_len, ylim[0] + y_len * 2, text_50)
    ax.text(xlim[0] + 2.5 * x_len, ylim[0] + y_len * 1, text_25)
    ax.text(xlim[0] + 2.5 * x_len, ylim[0] + y_len * 0, text_0)
    #
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * x_len],
        [ylim[0] + y_len * 0.25, ylim[0] + y_len * 0.25],
        "k:",
        linewidth=1,
    )
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * x_len],
        [ylim[0] + y_len * 1.25, ylim[0] + y_len * 1.25],
        "k--",
        linewidth=1,
    )
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * x_len],
        [ylim[0] + y_len * 2.25, ylim[0] + y_len * 2.25],
        "k-",
        linewidth=2,
    )
    ax.plot(
        [xlim[0], xlim[0] + 2.0 * x_len],
        [ylim[0] + y_len * 3.25, ylim[0] + y_len * 3.25],
        "k-",
        linewidth=4,
    )

    return fig


###############################################################################
##
##  APP
##
###############################################################################

WIDGET_WIDTH = "180px"
LEFT_PANEL_HEIGHT = "710px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "770px", 0]


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
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "by",
            "desc": "By Term:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "method",
            "desc": "Method:",
            "widget": widgets.Dropdown(
                options=["pearson", "kendall", "spearman"],
                ensure_option=True,
                continuous_update=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Frequency", "Cited_by"], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(5, 51, 5)), layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "min_link_value",
            "desc": "Min link value:",
            "widget": widgets.Dropdown(
                options="-1.00 -0.25 0.00 0.125 0.250 0.375 0.500 0.625 0.750 0.875".split(
                    " "
                ),
                ensure_option=True,
                continuous_update=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 6
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Heatmap", "Correlation map", "Chord diagram", "Table"],
                ensure_option=True,
                continuous_update=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 7
        {
            "arg": "cmap",
            "desc": "Matrix colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH), disabled=False,
            ),
        },
        # 8
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
        column = kwargs["term"]
        by = kwargs["by"]
        method = kwargs["method"]
        min_link_value = float(kwargs["min_link_value"].split(" ")[0])
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        view = kwargs["view"]
        sort_by = kwargs["sort_by"]
        layout = kwargs["layout"]
        figsize_width = int(kwargs["figsize_width"])
        figsize_height = int(kwargs["figsize_height"])
        #
        if view == "Matrix" or view == "Heatmap":
            controls[7]["widget"].disabled = False
            controls[8]["widget"].disabled = False
            controls[9]["widget"].disabled = True
        if view == "Correlation map":
            controls[7]["widget"].disabled = False
            controls[8]["widget"].disabled = True
            controls[9]["widget"].disabled = False
        if view == "Chord diagram":
            controls[7]["widget"].disabled = False
            controls[8]["widget"].disabled = True
            controls[9]["widget"].disabled = True
        #
        matrix = corr(
            x,
            column=column,
            by=by,
            method=method,
            cmap=cmap,
            top_by=top_by,
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
        )
        #
        #
        #
        if top_by == "Frequency":
            s = documents_by_term(x, column)
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[column].tolist(), s["Num_Documents"].tolist())
            }
        else:
            s = citations_by_term(x, column)
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[column].tolist(), s["Cited_by"].tolist())
            }
        matrix = matrix.rename(columns=new_names, index=new_names)
        output.clear_output()
        with output:
            if view == "Table" or view == "Heatmap":
                #
                # Sort order
                #
                if sort_by == "Frequency/Cited by asc.":
                    g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
                    names = sorted(matrix.columns, key=g, reverse=False)
                    matrix = matrix.loc[names, names]
                if sort_by == "Frequency/Cited by desc.":
                    g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
                    names = sorted(matrix.columns, key=g, reverse=True)
                    matrix = matrix.loc[names, names]
                if sort_by == "Alphabetic asc.":
                    matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
                        axis=1, ascending=True
                    )
                if sort_by == "Alphabetic desc.":
                    matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
                        axis=1, ascending=False
                    )
                #
                # View
                #
                with pd.option_context(
                    "display.max_columns", 60, "display.max_rows", 60
                ):
                    if view == "Table":
                        display(
                            matrix.style.format(
                                lambda q: "{:+4.3f}".format(q)
                                if q >= min_link_value
                                else ""
                            ).background_gradient(cmap=cmap)
                        )
                if view == "Heatmap":
                    display(
                        plt.heatmap(
                            matrix, cmap=cmap, figsize=(figsize_width, figsize_height)
                        )
                    )
                #
            if view == "Correlation map":
                #
                display(
                    correlation_map(
                        matrix=matrix,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                        min_link_value=min_link_value,
                    )
                )
                #
            if view == "Chord diagram":
                #
                display(
                    chord_diagram(
                        matrix,
                        figsize=(figsize_width, figsize_height),
                        cmap=cmap,
                        minval=min_link_value,
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


#
#
# APP
#
#
def app(df, limit_to=None, exclude=None):
    #
    body = widgets.Tab()
    body.children = [__TAB0__(df, limit_to, exclude)]
    body.set_title(0, "Matrix")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Correlation Analysis"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
