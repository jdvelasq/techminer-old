"""
Co-occurrence Analysis
==================================================================================================



"""
import ipywidgets as widgets
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.by_term import summary_by_term  # ,citations_by_term, documents_by_term,
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.keywords import Keywords
from techminer.plots import COLORMAPS
from techminer.params import EXCLUDE_COLS


def filter_index(
    x, column, matrix, axis, top_by=0, top_n=10, limit_to=None, exclude=None
):
    """
    Args:
        x: bibliographic database
        column: column analyzed
        matrix: result
        axis: 0=rows, 1=columns
        top_by: 0=frequency, 1=cited_by

    """
    top_terms = summary_by_term(x, column)

    if top_by == 0 or top_by == "Frequency":
        top_terms = top_terms.sort_values(
            ["Num_Documents", "Cited_by", column],
            ascending=[False, False, True],
            ignore_index=True,
        )

    if top_by == 1 or top_by == "Cited_by":
        top_terms = top_terms.sort_values(
            ["Cited_by", "Num_Documents", column],
            ascending=[False, False, True],
            ignore_index=True,
        )

    top_terms = top_terms[column]

    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        top_terms = top_terms[top_terms.map(lambda w: w in limit_to)]

    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        top_terms = top_terms[top_terms.map(lambda w: w not in exclude)]

    if top_n is not None:
        top_terms = top_terms.head(top_n)

    top_terms = top_terms.tolist()

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

    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1

    >>> compute_tfm(df, 'Author_Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1

    >>> compute_tfm(df, 'Authors', limit_to=['A', 'B'])
       A  B
    0  1  0
    1  1  1
    2  0  1
    3  1  1
    4  0  1

    >>> compute_tfm(df, 'Authors', exclude=['A', 'B'])
       C  D
    0  0  0
    1  0  0
    2  0  0
    3  1  0
    4  0  1


    """
    data = x[[column, "ID"]].copy()
    data["value"] = 1.0
    data = __explode(data, column)
    result = pd.pivot_table(
        data=data, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
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
    D  0  0  1

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


def occurrence(
    x,
    column,
    as_matrix=False,
    normalization=None,
    top_by=None,
    top_n=None,
    limit_to=None,
    exclude=None,
):
    """Computes the occurrence between the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (list, Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Cited_by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors  Cited_by  ID
    0       A         0   0
    1       A         1   1
    2     A;B         2   2
    3       B         3   3
    4   A;B;C         4   4
    5       D         5   5
    6     B;D         6   6

    >>> occurrence(df, column='Authors')
       A  B  C  D
    A  4  2  1  0
    B  2  4  1  1
    C  1  1  1  0
    D  0  1  0  2

    >>> occurrence(df, column='Authors', top_by='Frequency', top_n=3)
       A  B  D
    A  4  2  0
    B  2  4  1
    D  0  1  2

    >>> occurrence(df, column='Authors', limit_to=['A', 'B'])
       A  B
    A  4  2
    B  2  4

    >>> occurrence(df, column='Authors', exclude=['A', 'B'])
       C  D
    C  1  0
    D  0  2

    """
    return co_occurrence(
        x,
        column=column,
        by=None,
        top_by=top_by,
        top_n=top_n,
        limit_to=limit_to,
        exclude=exclude,
    )


def normalize_matrix(matrix, normalization=None):
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


#     # def correspondence_matrix(
#     #     self,
#     #     column_IDX,
#     #     column_COL,
#     #     sep_IDX=None,
#     #     sep_COL=None,
#     #     as_matrix=False,
#     #     keywords=None,
#     # ):
#     #     """

#     #     """
#     #     result = self.co_occurrence(
#     #         column_IDX=column_IDX,
#     #         column_COL=column_COL,
#     #         sep_IDX=sep_IDX,
#     #         sep_COL=sep_COL,
#     #         as_matrix=True,
#     #         minmax=None,
#     #         keywords=keywords,
#     #     )

#     #     matrix = result.transpose().values
#     #     grand_total = np.sum(matrix)
#     #     correspondence_matrix = np.divide(matrix, grand_total)
#     #     row_totals = np.sum(correspondence_matrix, axis=1)
#     #     col_totals = np.sum(correspondence_matrix, axis=0)
#     #     independence_model = np.outer(row_totals, col_totals)
#     #     norm_correspondence_matrix = np.divide(
#     #         correspondence_matrix, row_totals[:, None]
#     #     )
#     #     distances = np.zeros(
#     #         (correspondence_matrix.shape[0], correspondence_matrix.shape[0])
#     #     )
#     #     norm_col_totals = np.sum(norm_correspondence_matrix, axis=0)
#     #     for row in range(correspondence_matrix.shape[0]):
#     #         distances[row] = np.sqrt(
#     #             np.sum(
#     #                 np.square(
#     #                     norm_correspondence_matrix - norm_correspondence_matrix[row]
#     #                 )
#     #                 / col_totals,
#     #                 axis=1,
#     #             )
#     #         )
#     #     std_residuals = np.divide(
#     #         (correspondence_matrix - independence_model), np.sqrt(independence_model)
#     #     )
#     #     u, s, vh = np.linalg.svd(std_residuals, full_matrices=False)
#     #     deltaR = np.diag(np.divide(1.0, np.sqrt(row_totals)))
#     #     rowScores = np.dot(np.dot(deltaR, u), np.diag(s))

#     #     return pd.DataFrame(
#     #         data=rowScores, columns=result.columns, index=result.columns
#     #     )


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
LEFT_PANEL_HEIGHT = "655px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "720px", 0]

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
            "arg": "plot_type",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Table", "Heatmap", "Bubble plot", "Network"],
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
                options=["Frequency", "Cited_by"], layout=Layout(width=WIDGET_WIDTH),
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
            "arg": "normalization",
            "desc": "Normalization:",
            "widget": widgets.Dropdown(
                options=["None", "association", "inclusion", "jaccard", "salton"],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 8
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
        # 9
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 10
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
        term = kwargs["term"]
        by = kwargs["by"]
        plot_type = kwargs["plot_type"]
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        sort_by = kwargs["sort_by"]
        normalization = kwargs["normalization"]
        layout = kwargs["layout"]
        figsize_width = int(kwargs["figsize_width"])
        figsize_height = int(kwargs["figsize_height"])
        #
        matrix = co_occurrence(
            x,
            column=term,
            by=by,
            top_by=top_by,
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
        )
        #
        controls[7]["widget"].disabled = (
            False if plot_type == "Network" and term == by else True
        )
        controls[8]["widget"].disabled = False if plot_type == "Network" else True
        #
        s = summary_by_term(
            x=x,
            column=term,
            top_by=top_by,
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
        )
        new_names = {
            a: "{} [{:d}]".format(a, b)
            for a, b in zip(s[term].tolist(), s["Num_Documents"].tolist())
        }
        matrix = matrix.rename(columns=new_names)
        #
        if term == by:
            matrix = matrix.rename(index=new_names)
        else:
            s = summary_by_term(x, by)
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[by].tolist(), s["Num_Documents"].tolist())
            }
            matrix = matrix.rename(index=new_names)
        #
        # Sort order
        #
        g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
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
        if plot_type == "Network" and term == by:
            matrix = normalize_matrix(matrix, normalization)
        #
        output.clear_output()
        with output:
            #
            # View
            #
            if plot_type == "Table":
                with pd.option_context(
                    "display.max_columns", 50, "display.max_rows", 50
                ):
                    display(matrix.style.background_gradient(cmap=cmap, axis=None))
            if plot_type == "Heatmap":
                display(
                    plt.heatmap(
                        matrix, cmap=cmap, figsize=(figsize_width, figsize_height)
                    )
                )
            if plot_type == "Bubble plot":
                display(
                    plt.bubble(
                        matrix.transpose(),
                        axis=0,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if plot_type == "Network" and term == by:
                display(
                    occurrence_map(
                        matrix,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if plot_type == "Network" and term != by:
                display(
                    co_occurrence_map(
                        matrix,
                        layout=layout,
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
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
#  Occurrence Network
#
#


def app(df, limit_to=None, exclude=None):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__TAB0__(df, limit_to, exclude)]
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
