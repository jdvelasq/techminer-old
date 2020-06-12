"""
Factor analysis
==================================================================================================



"""
import networkx as nx
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from sklearn.decomposition import PCA
from techminer.by_term import summary_by_term
from techminer.co_occurrence import compute_tfm, most_cited_by, most_frequent
from techminer.keywords import Keywords
from techminer.plots import COLORMAPS


def factor_analysis(
    x, column, n_components=None, top_by=None, top_n=None, selected_columns=None
):
    """Computes the matrix of factors for terms in a given column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        n_components: Number of components to compute.
        as_matrix (bool): the result is reshaped by melt or not.
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
    ...       'Author Keywords': y,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author Keywords  Cited by  ID
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


    >>> factor_analysis(df, 'Authors', n_components=3)
             F0            F1       F2
    A -0.774597 -0.000000e+00  0.00000
    B  0.258199  7.071068e-01 -0.57735
    C -0.258199  7.071068e-01  0.57735
    D  0.516398  1.110223e-16  0.57735

    

    >>> factor_analysis(df, 'Authors', n_components=3, keywords=['A', 'B', 'C'])
             F0        F1        F2
    A -0.888074  0.000000  0.459701
    B  0.325058  0.707107  0.627963
    C -0.325058  0.707107 -0.627963

    """

    tfm = compute_tfm(x, column, selected_columns)
    terms = tfm.columns.tolist()
    if n_components is None:
        n_components = int(np.sqrt(len(set(terms))))
    pca = PCA(n_components=n_components)
    result = np.transpose(pca.fit(X=tfm.values).components_)
    result = pd.DataFrame(
        result, columns=["F" + str(i) for i in range(n_components)], index=terms
    )

    if selected_columns is None:
        if (top_by == 0 or top_by == "Frequency") and top_n is not None:
            selected_columns = set(most_frequent(x, column, top_n))
        if (top_by == 1 or top_by == "Cited_by") and top_n is not None:
            selected_columns = set(most_cited_by(x, column, top_n))
        if selected_columns is not None:
            selected_columns = list(selected_columns)

    if selected_columns is not None:
        result = result.loc[selected_columns, :]

    return result


def factor_map(
    matrix, layout="Kamada Kawai", cmap="Greys", figsize=(17, 12), min_link_value=0
):
    """
    """

    #
    # Data preparation
    #
    terms = matrix.index.tolist()

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
    for column in matrix.columns:

        matrix = matrix.sort_values(column, ascending=False)

        x = matrix[column][matrix[column] >= 0.25]
        if len(x) > 1:
            for t0 in range(len(x.index) - 1):
                for t1 in range(1, len(x.index)):
                    value = 0.5 * (x[t0] + x[t1])
                    if value >= 0.75:
                        G.add_edge(x.index[t0], x.index[t1], width=3)
                    elif value >= 0.50:
                        G.add_edge(x.index[t0], x.index[t1], width=2)
                    elif value >= 0.25:
                        G.add_edge(x.index[t0], x.index[t1], width=1)
        #
        x = matrix[column][matrix[column] < -0.25]
        if len(x) > 1:
            for t0 in range(len(x.index) - 1):
                for t1 in range(1, len(x.index)):
                    value = 0.5 * (x[t0] + x[t1])
                    if value <= -0.75:
                        G.add_edge(x.index[t0], x.index[t1], width=3)
                    elif value <= -0.50:
                        G.add_edge(x.index[t0], x.index[t1], width=2)
                    elif value <= -0.25:
                        G.add_edge(x.index[t0], x.index[t1], width=1)

    #
    # network edges
    #
    for e in G.edges.data():
        a, b, dic = e
        edge = [(a, b)]
        draw(
            G,
            ax=ax,
            edgelist=edge,
            width=dic["width"],
            edge_color="k",
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
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

    ax.set_xlim(
        xlim[0] - 0.15 * (xlim[1] - xlim[0]), xlim[1] + 0.15 * (xlim[1] - xlim[0])
    )
    ax.set_ylim(
        ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0])
    )
    ax.set_aspect("equal")
    return fig


#
#
# APP
#
#

WIDGET_WIDTH = "200px"
LEFT_PANEL_HEIGHT = "588px"
RIGHT_PANEL_WIDTH = "870px"
FIGSIZE = (14, 10.0)
PANE_HEIGHTS = ["80px", "650px", 0]

COLUMNS = [
    "Author Keywords",
    "Authors",
    "Countries",
    "Index Keywords",
    "Institutions",
    "Keywords",
]


def __TAB0__(x):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "n_components",
            "desc": "Number of factors:",
            "widget": widgets.Dropdown(
                options=list(range(2, 21)),
                value=2,
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
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
            "widget": widgets.IntSlider(
                value=5,
                min=5,
                max=50,
                step=1,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=[
                    "Alphabetic asc.",
                    "Alphabetic desc.",
                    "Frequency/Cited by asc.",
                    "Frequency/Cited by desc.",
                    "Factor asc.",
                    "Factor desc.",
                ],
                disable=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 6
        {
            "arg": "factor",
            "desc": "Sort by factor:",
            "widget": widgets.Dropdown(
                options=[], disable=True, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 7
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Matrix", "Network"],
                disable=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        #  8
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
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        term = kwargs["term"]
        n_components = int(kwargs["n_components"])
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        sort_by = kwargs["sort_by"]
        factor = kwargs["factor"]
        view = kwargs["view"]
        layout = kwargs["layout"]
        #
        matrix = factor_analysis(
            x=x,
            column=term,
            n_components=n_components,
            top_by=top_by,
            top_n=top_n,
            selected_columns=None,
        )
        #
        controls[6]["widget"].options = matrix.columns
        controls[6]["widget"].disabled = (
            True if sort_by not in ["Factor asc.", "Factor desc."] else False
        )
        controls[8]["widget"].disabled = False if view == "Network" else True
        #
        s = summary_by_term(x, term)
        new_names = {
            a: "{} [{:d}]".format(a, b)
            for a, b in zip(s[term].tolist(), s["Num_Documents"].tolist())
        }
        matrix = matrix.rename(index=new_names)
        #
        g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
        if sort_by == "Frequency/Cited by asc.":
            names = sorted(matrix.index, key=g, reverse=False)
            matrix = matrix.loc[names, :]
        if sort_by == "Frequency/Cited by desc.":
            names = sorted(matrix.index, key=g, reverse=True)
            matrix = matrix.loc[names, :]
        if sort_by == "Alphabetic asc.":
            matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
                axis=1, ascending=True
            )
        if sort_by == "Alphabetic desc.":
            matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
                axis=1, ascending=False
            )
        if sort_by == "Factor asc.":
            matrix = matrix.sort_values(factor, ascending=True)
        if sort_by == "Factor desc.":
            matrix = matrix.sort_values(factor, ascending=False)
        #
        output.clear_output()
        with output:
            if view == "Matrix":
                display(
                    matrix.style.format(
                        lambda q: "{:+4.3f}".format(q)
                    ).background_gradient(cmap=cmap, axis=None)
                )
            else:
                display(factor_map(matrix, layout=layout, cmap=cmap, figsize=(17, 12)))

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
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


def app(df):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__TAB0__(df)]
    body.set_title(0, "Matrix")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Factor Analysis"
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
