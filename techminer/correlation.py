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
import techminer.common as cmn
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
from techminer.document_term import TF_matrix

from matplotlib.lines import Line2D

###################################################################################################
##
##  CALCULATIONS
##
###################################################################################################


def correlation_matrix(
    x,
    column,
    by=None,
    method="pearson",
    top_by=None,
    top_n=None,
    limit_to=None,
    exclude=None,
    sort_c_by="Num Documents",
    c_ascending=False,
    sort_r_by="Num Documents",
    r_ascending=False,
):

    x = x.copy()

    if by is None:
        by = column

    if column == by:

        A = TF_matrix(x, column=column)
        A = cmn.limit_to_exclude(
            data=A, axis=1, column=column, limit_to=limit_to, exclude=exclude,
        )
        A = cmn.add_counters_to_axis(X=A, axis=1, data=x, column=column)
        A = cmn.sort_by_axis(data=A, sort_by=top_by, ascending=False, axis=1)
        A = A[A.columns[:top_n]]
        matrix = A.corr(method=method)

    else:

        w = x[[column, by, "ID"]].dropna()

        A = TF_matrix(w, column=column)
        A = cmn.limit_to_exclude(
            data=A, axis=1, column=column, limit_to=limit_to, exclude=exclude,
        )
        A = cmn.add_counters_to_axis(X=A, axis=1, data=x, column=column)
        A = cmn.sort_by_axis(data=A, sort_by=top_by, ascending=False, axis=1)
        A = A[A.columns[:top_n]]

        B = TF_matrix(w, column=by)
        matrix = np.matmul(B.transpose().values, A.values)
        matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)
        matrix = matrix.corr(method=method)

    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_r_by, ascending=r_ascending, axis=0
    )

    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_c_by, ascending=c_ascending, axis=1
    )

    return matrix


# -------------------------------------------------------------------------------------------------


def chord_diagram(
    matrix, cmap="Greys", figsize=(17, 12),
):
    x = matrix.copy()
    terms = matrix.columns.tolist()
    node_sizes = cmn.counters_to_node_sizes(x=terms)
    node_colors = cmn.counters_to_node_colors(x, cmap=pyplot.cm.get_cmap(cmap))

    cd = ChordDiagram()

    ## add nodes
    for idx, term in enumerate(x.columns):
        cd.add_node(term, color=node_colors[idx], s=node_sizes[idx])

    ## add links
    m = x.stack().to_frame().reset_index()
    m = m[m.level_0 < m.level_1]
    m.columns = ["from_", "to_", "link_"]
    m = m.reset_index(drop=True)

    d = {
        0: {"linestyle": "-", "linewidth": 4, "color": "black"},
        1: {"linestyle": "-", "linewidth": 2, "color": "black"},
        2: {"linestyle": "--", "linewidth": 1, "color": "gray"},
        3: {"linestyle": ":", "linewidth": 1, "color": "lightgray"},
    }

    for idx in range(len(m)):

        key = (
            0
            if m.link_[idx] > 0.75
            else (1 if m.link_[idx] > 0.50 else (2 if m.link_[idx] > 0.25 else 3))
        )

        cd.add_edge(m.from_[idx], m.to_[idx], **(d[key]))

    return cd.plot(figsize=figsize)


# -------------------------------------------------------------------------------------------------


def correlation_map(
    X, layout="Kamada Kawai", iterations=50, cmap="Greys", figsize=(17, 12),
):
    """Computes the correlation map directly using networkx.
    """

    if len(X.columns) > 50:
        return "Maximum number of nodes exceded!"

    ## Networkx
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    ## Data preparation
    terms = X.columns.tolist()
    node_sizes = cmn.counters_to_node_sizes(x=terms)
    node_colors = cmn.counters_to_node_colors(x=terms, cmap=pyplot.cm.get_cmap(cmap))

    ## Add nodes
    G.add_nodes_from(terms)

    ## node positions
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

    ## links
    m = X.stack().to_frame().reset_index()
    m = m[m.level_0 < m.level_1]
    m.columns = ["from_", "to_", "link_"]
    m = m[m.link_ > 0.0]
    m = m.reset_index(drop=True)

    d = {
        0: {"width": 4, "style": "solid", "edge_color": "k"},
        1: {"width": 2, "style": "solid", "edge_color": "k"},
        2: {"width": 1, "style": "dashed", "edge_color": "gray"},
        3: {"width": 1, "style": "dotted", "edge_color": "gray"},
    }

    n_edges_0 = 0
    n_edges_25 = 0
    n_edges_50 = 0
    n_edges_75 = 0

    for idx in range(len(m)):

        edge = [(m.from_[idx], m.to_[idx])]
        key = (
            0
            if m.link_[idx] > 0.75
            else (1 if m.link_[idx] > 0.50 else (2 if m.link_[idx] > 0.25 else 3))
        )

        n_edges_75 += 1 if m.link_[idx] >= 0.75 else 0
        n_edges_50 += 1 if m.link_[idx] >= 0.50 and m.link_[idx] < 0.75 else 0
        n_edges_25 += 1 if m.link_[idx] >= 0.25 and m.link_[idx] < 0.50 else 0
        n_edges_0 += 1 if m.link_[idx] > 0 and m.link_[idx] < 0.25 else 0

        nx.draw_networkx_edges(
            G, pos=pos, ax=ax, node_size=1, with_labels=False, edgelist=edge, **(d[key])
        )

    ## nodes
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        edge_color="k",
        nodelist=terms,
        node_size=node_sizes,
        node_color=node_colors,
        node_shape="o",
        edgecolors="k",
        linewidths=1,
    )

    ## node labels
    cmn.ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)

    ## Figure size
    cmn.ax_expand_limits(ax)

    ##
    legend_lines = [
        Line2D([0], [0], color="k", linewidth=4, linestyle="-"),
        Line2D([0], [0], color="k", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="gray", linewidth=1, linestyle="--"),
        Line2D([0], [0], color="gray", linewidth=1, linestyle=":"),
    ]

    text_75 = "> 0.75 ({})".format(n_edges_75)
    text_50 = "0.50-0.75 ({})".format(n_edges_50)
    text_25 = "0.25-0.50 ({})".format(n_edges_25)
    text_0 = "< 0.25 ({})".format(n_edges_0)

    ax.legend(legend_lines, [text_75, text_50, text_25, text_0])

    ax.axis("off")

    return fig


###################################################################################################
##
##  APP
##
###################################################################################################


def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Correlation Analysis",
        tab_titles=["Correlation"],
        tab_widgets=[TABapp0(data, limit_to=limit_to, exclude=exclude).run(),],
        tab=tab,
    )


###################################################################################################
##
##  TAB app 0 --- Correlation analysis
##
###################################################################################################


class TABapp0(gui.TABapp_):
    def __init__(self, data, limit_to, exclude):

        super(TABapp0, self).__init__()

        self.data_ = data
        self.limit_to_ = limit_to
        self.exclude_ = exclude

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_ = [
            gui.dropdown(
                desc="View:",
                options=[
                    "Matrix",
                    "Heatmap",
                    "Bubble plot",
                    "Correlation map",
                    "Chord diagram",
                ],
            ),
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(
                desc="By:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(desc="Method:", options=["pearson", "kendall", "spearman"],),
            gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            gui.top_n(),
            gui.dropdown(
                desc="Sort C-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            gui.c_axis_ascending(),
            gui.dropdown(
                desc="Sort R-axis by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            gui.r_axis_ascending(),
            gui.cmap(),
            gui.nx_layout(),
            gui.nx_max_iter(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def gui(self, **kwargs):

        super().gui(**kwargs)

        if self.panel_[1]["widget"].value == self.panel_[2]["widget"].value:
            self.grid_[0, 0].disabled = True
            self.panel_[0]["widget"].disabled = True
            for i in range(3, len(self.panel_)):
                self.panel_[i]["widget"].disabled = True
            return

        self.grid_[0, 0].disabled = False
        for i in range(len(self.panel_)):
            self.panel_[i]["widget"].disabled = False

        for i in [-9, -8, -7, -6]:
            self.panel_[i]["widget"].disabled = self.view not in [
                "Matrix",
                "Heatmap",
                "Bubble plot",
            ]

        self.panel_[-4]["widget"].disabled = self.view != "Correlation map"
        self.panel_[-3]["widget"].disabled = self.view != "Correlation map"
        self.panel_[-2]["widget"].disabled = self.view == "Matrix"
        self.panel_[-1]["widget"].disabled = self.view == "Matrix"

    def update(self, button):
        """ 
        """

        self.output_.clear_output()
        with self.output_:
            display(gui.processing())

        self.matrix_ = correlation_matrix(
            x=self.data_,
            column=self.column,
            by=self.by,
            method=self.method,
            top_by=self.top_by,
            top_n=self.top_n,
            limit_to=self.limit_to_,
            exclude=self.exclude_,
            sort_c_by=self.sort_c_axis_by,
            c_ascending=self.c_axis_ascending,
            sort_r_by=self.sort_r_axis_by,
            r_ascending=self.r_axis_ascending,
        )

        self.output_.clear_output()
        with self.output_:

            if self.matrix_ is None:
                display(widgets.HTML("Different columns must be selected!"))
                return

            if self.view == "Matrix":
                display(
                    self.matrix_.style.format("{:+4.3f}").background_gradient(
                        cmap=self.cmap, axis=None
                    )
                )

            if self.view == "Heatmap":
                display(
                    plt.heatmap(
                        self.matrix_, cmap=self.cmap, figsize=(self.width, self.height)
                    )
                )

            if self.view == "Bubble plot":
                display(
                    plt.bubble(
                        self.matrix_,
                        axis=0,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )

            if self.view == "Correlation map":
                display(
                    correlation_map(
                        X=self.matrix_,
                        layout=self.layout,
                        iterations=self.nx_max_iter,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )

            if self.view == "Chord diagram":
                display(
                    chord_diagram(
                        matrix=self.matrix_,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )


##################################################################################################


###############################################################################
##
##  APP
##
###############################################################################


# def __TAB0__(x, limit_to, exclude):
#     # -------------------------------------------------------------------------
#     #
#     # UI
#     #
#     # -------------------------------------------------------------------------
#     COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
#     #
#     left_panel = [
#         gui.dropdown(
#             desc="View:",
#             options=["Matrix", "Heatmap", "Correlation map", "Chord diagram"],
#         ),
#         gui.dropdown(desc="Term:", options=[z for z in COLUMNS if z in x.columns],),
#         gui.dropdown(desc="By:", options=[z for z in COLUMNS if z in x.columns],),
#         gui.dropdown(desc="Method:", options=["pearson", "kendall", "spearman"],),
#         gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
#         gui.top_n(),
#         gui.dropdown(
#             desc="Sort by:", options=["Alphabetic", "Num Documents", "Times Cited",],
#         ),
#         gui.ascending(),
#         gui.dropdown(desc="Min link value:", options="0.00 0.25 0.50 0.75".split(" "),),
#         gui.cmap(),
#         gui.nx_layout(),
#         gui.fig_width(),
#         gui.fig_height(),
#     ]
#     # -------------------------------------------------------------------------
#     #
#     # Logic
#     #
#     # -------------------------------------------------------------------------
#     def server(**kwargs):
#         #
#         column = kwargs["term"]
#         by = kwargs["by"]
#         method = kwargs["method"]
#         min_link_value = float(kwargs["min_link_value"].split(" ")[0])
#         cmap = kwargs["cmap"]
#         top_by = kwargs["top_by"]
#         top_n = int(kwargs["top_n"])
#         view = kwargs["view"]
#         sort_by = kwargs["sort_by"]
#         ascending = kwargs["ascending"]
#         layout = kwargs["layout"]
#         width = int(kwargs["width"])
#         height = int(kwargs["height"])
#         #
#         if view == "Matrix" or view == "Heatmap":
#             left_panel[-7]["widget"].disabled = False
#             left_panel[-6]["widget"].disabled = False
#             left_panel[-5]["widget"].disabled = False
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = True
#             left_panel[-1]["widget"].disabled = True
#         if view == "Heatmap":
#             left_panel[-7]["widget"].disabled = False
#             left_panel[-6]["widget"].disabled = False
#             left_panel[-5]["widget"].disabled = False
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False
#         if view == "Correlation map":
#             left_panel[-7]["widget"].disabled = True
#             left_panel[-6]["widget"].disabled = True
#             left_panel[-5]["widget"].disabled = True
#             left_panel[-3]["widget"].disabled = False
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False
#         if view == "Chord diagram":
#             left_panel[-7]["widget"].disabled = True
#             left_panel[-6]["widget"].disabled = True
#             left_panel[-5]["widget"].disabled = True
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False
#         #

#         #
#         output.clear_output()
#         with output:
#             display(
#                 corr(
#                     x,
#                     column=column,
#                     by=by,
#                     method=method,
#                     output=view,
#                     cmap=cmap,
#                     top_by=top_by,
#                     top_n=top_n,
#                     sort_by=sort_by,
#                     ascending=ascending,
#                     layout=layout,
#                     limit_to=limit_to,
#                     exclude=exclude,
#                     figsize=(width, height),
#                 )
#             )

#         return
#         #
#         #
#         #
#         # if top_by == "Num Documents":
#         #     s = summary_by_term(x, column)
#         #     new_names = {
#         #         a: "{} [{:d}]".format(a, b)
#         #         for a, b in zip(s[column].tolist(), s["Num_Documents"].tolist())
#         #     }
#         # else:
#         #     s = summary_by_term(x, column)
#         #     new_names = {
#         #         a: "{} [{:d}]".format(a, b)
#         #         for a, b in zip(s[column].tolist(), s["Times_Cited"].tolist())
#         #     }
#         # matrix = matrix.rename(columns=new_names, index=new_names)

#         # output.clear_output()
#         # with output:
#         #     if view == "Matrix" or view == "Heatmap":
#         #         #
#         #         # Sort order
#         #         #
#         #         g = (
#         #             lambda m: m[m.find("[") + 1 : m.find("]")].zfill(5)
#         #             + " "
#         #             + m[: m.find("[") - 1]
#         #         )
#         #         if sort_by == "Frequency/Cited by asc.":
#         #             names = sorted(matrix.columns, key=g, reverse=False)
#         #             matrix = matrix.loc[names, names]
#         #         if sort_by == "Frequency/Cited by desc.":
#         #             names = sorted(matrix.columns, key=g, reverse=True)
#         #             matrix = matrix.loc[names, names]
#         #         if sort_by == "Alphabetic asc.":
#         #             matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
#         #                 axis=1, ascending=True
#         #             )
#         #         if sort_by == "Alphabetic desc.":
#         #             matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
#         #                 axis=1, ascending=False
#         #             )
#         #         #
#         #         # View
#         #         #
#         #         with pd.option_context(
#         #             "display.max_columns", 60, "display.max_rows", 60
#         #         ):
#         #             if view == "Matrix":
#         #                 display(
#         #                     matrix.style.format(
#         #                         lambda q: "{:+4.3f}".format(q)
#         #                         if q >= min_link_value
#         #                         else ""
#         #                     ).background_gradient(cmap=cmap)
#         #                 )
#         #         if view == "Heatmap":
#         #             display(
#         #                 plt.heatmap(
#         #                     matrix, cmap=cmap, figsize=(figsize_width, figsize_height)
#         #                 )
#         #             )
#         #         #
#         #     if view == "Correlation map":
#         #         #
#         #         display(
#         #             correlation_map(
#         #                 matrix=matrix,
#         #                 summary=summary_by_term(
#         #                     x, column=column, top_by=None, top_n=None
#         #                 ),
#         #                 layout=layout,
#         #                 cmap=cmap,
#         #                 figsize=(figsize_width, figsize_height),
#         #                 min_link_value=min_link_value,
#         #             )
#         #         )
#         #         #
#         #     if view == "Chord diagram":
#         #         #
#         #         display(
#         #             chord_diagram(
#         #                 matrix,
#         #                 figsize=(figsize_width, figsize_height),
#         #                 cmap=cmap,
#         #                 minval=min_link_value,
#         #             )
#         #         )

#     ###
#     output = widgets.Output()
#     return gui.TABapp(left_panel=left_panel, server=server, output=output)


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
