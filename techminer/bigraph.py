"""
Co-occurrence Analysis
==================================================================================================



"""
import warnings

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from techminer.document_term import TF_matrix

import techminer.by_term as by_term
import techminer.plots as plt
from techminer.chord_diagram import ChordDiagram
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.keywords import Keywords
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS
import techminer.common as cmn
import techminer.gui as gui

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################################
##
##  CALCULATIONS
##
################################################################################################


def co_occurrence_matrix(
    x,
    column,
    by=None,
    top_by=None,
    top_n=None,
    limit_to=None,
    exclude=None,
    sort_c_by="Num Documents",
    c_ascending=False,
    sort_r_by="Num Documents",
    r_ascending=False,
):
    """
    """
    if column == by:
        x = None
        return

    W = x[[column, by, "ID"]].dropna()
    A = TF_matrix(W, column)
    B = TF_matrix(W, by)

    if top_by == "Data":

        A = cmn.limit_to_exclude(
            data=A, axis=1, column=column, limit_to=limit_to, exclude=exclude,
        )
        B = cmn.limit_to_exclude(
            data=B, axis=1, column=by, limit_to=limit_to, exclude=exclude,
        )
        matrix = np.matmul(B.transpose().values, A.values)
        matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)

        # sort max values per column
        max_columns = matrix.sum(axis=0)
        max_columns = max_columns.sort_values(ascending=False)
        max_columns = max_columns.head(top_n).index

        max_index = matrix.sum(axis=1)
        max_index = max_index.sort_values(ascending=False)
        max_index = max_index.head(top_n).index

        matrix = matrix.loc[
            [t for t in matrix.index if t in max_index],
            [t for t in matrix.columns if t in max_columns],
        ]

        matrix = cmn.add_counters_to_axis(X=matrix, axis=1, data=x, column=column)
        matrix = cmn.add_counters_to_axis(X=matrix, axis=0, data=x, column=by)

    if top_by in ["Num Documents", "Times Cited"]:

        A = cmn.limit_to_exclude(
            data=A, axis=1, column=column, limit_to=limit_to, exclude=exclude,
        )

        A = cmn.add_counters_to_axis(X=A, axis=1, data=x, column=column)

        A = cmn.sort_by_axis(data=A, sort_by=top_by, ascending=False, axis=1)

        A = A[A.columns[:top_n]]

        B = cmn.limit_to_exclude(
            data=B, axis=1, column=by, limit_to=limit_to, exclude=exclude,
        )

        B = cmn.add_counters_to_axis(X=B, axis=1, data=x, column=by)

        B = cmn.sort_by_axis(data=B, sort_by=top_by, ascending=False, axis=1)
        B = B[B.columns[:top_n]]

        matrix = np.matmul(B.transpose().values, A.values)
        matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)

    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_r_by, ascending=r_ascending, axis=0
    )

    matrix = cmn.sort_by_axis(
        data=matrix, sort_by=sort_c_by, ascending=c_ascending, axis=1
    )

    return matrix


# ----------------------------------------------------------------------------------------------


def slope_chart(matrix, figsize, cmap_column="Greys", cmap_by="Greys"):
    """
    """
    matplotlib.rc("font", size=12)

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap_column = pyplot.cm.get_cmap(cmap_column)
    cmap_by = pyplot.cm.get_cmap(cmap_by)

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
    # Sizes
    #
    left_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.index]
    right_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.columns]

    min_size = min(left_sizes + right_sizes)
    max_size = max(left_sizes + right_sizes)

    left_sizes = [
        150 + 2000 * (t - min_size) / (max_size - min_size) for t in left_sizes
    ]
    right_sizes = [
        150 + 2000 * (t - min_size) / (max_size - min_size) for t in right_sizes
    ]

    #
    # Colors
    #
    left_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.index]
    right_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.columns]

    min_color = min(left_colors + right_colors)
    max_color = max(left_colors + right_colors)

    left_colors = [
        cmap_by(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
        for t in left_colors
    ]
    right_colors = [
        cmap_column(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
        for t in right_colors
    ]

    ax.scatter(
        x=[1] * m,
        y=yleft,
        s=left_sizes,
        c=left_colors,
        zorder=10,
        linewidths=1,
        edgecolors="k",
    )

    for idx, text in enumerate(matrix.index):
        ax.plot([0.7, 1.0], [yleft[idx], yleft[idx]], "-", c="grey")

    for idx, text in enumerate(matrix.index):
        ax.text(
            0.7,
            yleft[idx],
            text,
            fontsize=10,
            ha="right",
            va="center",
            zorder=10,
            bbox=dict(
                facecolor="w", alpha=1.0, edgecolor="gray", boxstyle="round,pad=0.5",
            ),
        )

    #
    # right y-axis
    #

    ax.scatter(
        x=[3] * n,
        y=yright,
        s=right_sizes,
        c=right_colors,
        zorder=10,
        linewidths=1,
        edgecolors="k",
    )

    for idx, text in enumerate(matrix.columns):
        ax.plot([3.0, 3.3], [yright[idx], yright[idx]], "-", c="grey")

    for idx, text in enumerate(matrix.columns):
        ax.text(
            3.3,
            yright[idx],
            text,
            fontsize=10,
            ha="left",
            va="center",
            bbox=dict(
                facecolor="w", alpha=1.0, edgecolor="gray", boxstyle="round,pad=0.5",
            ),
            zorder=11,
        )

    #
    # Figure size
    #
    cmn.ax_expand_limits(ax)
    ax.invert_yaxis()
    ax.axis("off")

    return fig


# ----------------------------------------------------------------------------------------------


def co_occurrence_map(
    X,
    layout="Kamada Kawai",
    iterations=50,
    cmap="Greys",
    cmap_by="Greys",
    figsize=(17, 12),
):
    """Computes the occurrence map directly using networkx.
    """

    ## Networkx
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    ## Data preparation
    terms = X.columns.tolist() + X.index.tolist()

    node_sizes = cmn.counters_to_node_sizes(x=terms)
    column_node_sizes = node_sizes[: len(X.index)]
    index_node_sizes = node_sizes[len(X.index) :]

    node_colors = cmn.counters_to_node_colors(x=terms, cmap=lambda w: w)
    column_node_colors = node_colors[: len(X.index)]
    index_node_colors = node_colors[len(X.index) :]

    cmap = pyplot.cm.get_cmap(cmap)
    cmap_by = pyplot.cm.get_cmap(cmap_by)

    index_node_colors = [cmap_by(t) for t in index_node_colors]
    column_node_colors = [cmap(t) for t in column_node_colors]

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
    m.columns = ["from_", "to_", "link_"]
    m = m[m.link_ > 0.0]
    m = m.reset_index(drop=True)

    max_width = m.link_.max()
    for idx in range(len(m)):

        edge = [(m.from_[idx], m.to_[idx])]
        width = 0.2 + 3.8 * m.link_[idx] / max_width
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            node_size=1,
            with_labels=False,
            edge_color="k",
            edgelist=edge,
            width=width,
        )

    #
    # Draw column nodes
    #
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        edge_color="k",
        nodelist=X.columns.tolist(),
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
        nodelist=X.index.tolist(),
        node_size=index_node_sizes,
        node_color=index_node_colors,
        node_shape="o",
        edgecolors="k",
        linewidths=1,
    )

    node_sizes = column_node_sizes + index_node_sizes
    cmn.ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)
    cmn.ax_expand_limits(ax)
    ax.set_aspect("equal")
    ax.axis("off")
    cmn.set_ax_splines_invisible(ax)
    return fig


################################################################################################
##
##  APP
##
################################################################################################


def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Bigraph Analysis",
        tab_titles=["Network Map", "Associations Map", "OLD"],
        tab_widgets=[
            TABapp0(data, limit_to=limit_to, exclude=exclude).run(),
            #  TABapp1(data, limit_to=limit_to, exclude=exclude).run(),
            #  __TAB1__(data, limit_to=limit_to, exclude=exclude),
        ],
        tab=tab,
    )


################################################################################################
##
##  TAB app 0 --- Bigraph
##
################################################################################################


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
                options=["Matrix", "Heatmap", "Bubble plot", "Network", "Slope chart",],
            ),
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(
                desc="By:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(
                desc="Top by:", options=["Num Documents", "Times Cited", "Data",],
            ),
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
            gui.cmap(arg="cmap", desc="Colormap Col:"),
            gui.cmap(arg="cmap_by", desc="Colormap By:"),
            gui.nx_layout(),
            gui.nx_max_iter(),
            gui.fig_width(),
            gui.fig_height(),
            # {
            #     "arg": "terms",
            #     "desc": "Terms:",
            #     "widget": widgets.widgets.SelectMultiple(
            #         options=[], layout=Layout(width="98%", height="180px"),
            #     ),
            # },
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

        self.panel_[5]["widget"].disabled = self.view == "Network"
        self.panel_[6]["widget"].disabled = self.view == "Network"
        self.panel_[7]["widget"].disabled = self.view == "Network"
        self.panel_[8]["widget"].disabled = self.view == "Network"

        self.panel_[-5]["widget"].disabled = self.view not in ["Network", "Slope chart"]
        self.panel_[-4]["widget"].disabled = self.view != "Network"
        self.panel_[-3]["widget"].disabled = self.view != "Network"
        self.panel_[-2]["widget"].disabled = self.view == "Matrix"
        self.panel_[-1]["widget"].disabled = self.view == "Matrix"

    def update(self, button):
        """ 
        """

        self.output_.clear_output()
        with self.output_:
            display(gui.processing())

        self.matrix_ = co_occurrence_matrix(
            x=self.data_,
            column=self.column,
            by=self.by,
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
                    self.matrix_.style.background_gradient(cmap=self.cmap, axis=None)
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

            if self.view == "Slope chart":
                display(
                    slope_chart(
                        self.matrix_,
                        figsize=(self.width, self.height),
                        cmap_column=self.cmap,
                        cmap_by=self.cmap_by,
                    )
                )

            if self.view == "Network":
                display(
                    co_occurrence_map(
                        X=self.matrix_,
                        layout=self.layout,
                        iterations=self.nx_max_iter,
                        cmap=self.cmap,
                        cmap_by=self.cmap_by,
                        figsize=(self.width, self.height),
                    )
                )

            # if self.view == "Table":

            #     result = self.matrix_.stack().to_frame().reset_index()
            #     result.columns = [self.by, self.column, "Values"]

            #     result = result[result["Values"] != 0]
            #     result = result.sort_values(["Values"])
            #     result = result.reset_index(drop=True)

            #     if self.sort_by == "Alphabetic":
            #         result = result.sort_values(
            #             [by, column, "Values"], ascending=self.ascending
            #         )

            #     if self.sort_by == "Num Documents":
            #         result["ND-column"] = result[column].map(
            #             lambda w: w.split(" ")[-1].split(":")[0]
            #         )
            #         result["ND-by"] = result[by].map(
            #             lambda w: w.split(" ")[-1].split(":")[0]
            #         )
            #         result = result.sort_values(
            #             ["ND-by", "ND-column", "Values"], ascending=self.ascending
            #         )
            #         result.pop("ND-column")
            #         result.pop("ND-by")

            #     display(result)


###############################################################################
##
##  TAB app 1 --- Associations
##
###############################################################################


# class TABapp1(gui.TABapp_):
#     def __init__(self, data, limit_to, exclude):

#         super(TABapp1, self).__init__()

#         self.data_ = data
#         self.limit_to_ = limit_to
#         self.exclude_ = exclude

#         COLUMNS = sorted(
#             [column for column in data.columns if column not in EXCLUDE_COLS]
#         )

#         self.panel_ = [
#             gui.dropdown(desc="View:", options=["Table", "Network",],),
#             gui.dropdown(
#                 desc="Column:", options=[z for z in COLUMNS if z in data.columns],
#             ),
#             gui.dropdown(
#                 desc="By:", options=[z for z in COLUMNS if z in data.columns],
#             ),
#             gui.dropdown(
#                 desc="Top by:", options=["Num Documents", "Times Cited", "Values",],
#             ),
#             gui.top_n(),
#             gui.dropdown(
#                 desc="Sort by:",
#                 options=["Alphabetic", "Num Documents", "Times Cited",],
#             ),
#             gui.ascending(),
#             gui.cmap(arg="cmap", desc="Colormap Col:"),
#             gui.cmap(arg="cmap_by", desc="Colormap By:"),
#             gui.nx_layout(),
#             gui.fig_width(),
#             gui.fig_height(),
#             {
#                 "arg": "terms",
#                 "desc": "Terms:",
#                 "widget": widgets.widgets.SelectMultiple(
#                     options=[], layout=Layout(width="98%", height="100px"),
#                 ),
#             },
#         ]
#         super().create_grid()

#     def gui(self, **kwargs):

#         super().gui(**kwargs)

#         if self.view in ["Matrix", "Table"]:
#             self.panel_[-4]["widget"].disabled = True
#             self.panel_[-3]["widget"].disabled = True
#             self.panel_[-2]["widget"].disabled = True
#             self.panel_[-1]["widget"].disabled = True

#         if self.view == "Heatmap":
#             self.panel_[-4]["widget"].disabled = True
#             self.panel_[-3]["widget"].disabled = True
#             self.panel_[-2]["widget"].disabled = False
#             self.panel_[-1]["widget"].disabled = False

#         if self.view == "Bubble plot":
#             self.panel_[-4]["widget"].disabled = True
#             self.panel_[-3]["widget"].disabled = True
#             self.panel_[-2]["widget"].disabled = False
#             self.panel_[-1]["widget"].disabled = False

#         if self.view == "Network":
#             self.panel_[-4]["widget"].disabled = False
#             self.panel_[-3]["widget"].disabled = False
#             self.panel_[-2]["widget"].disabled = False
#             self.panel_[-1]["widget"].disabled = False

#         if self.view == "Slope chart":
#             self.panel_[-4]["widget"].disabled = False
#             self.panel_[-3]["widget"].disabled = True
#             self.panel_[-2]["widget"].disabled = False
#             self.panel_[-1]["widget"].disabled = False

#     def update(self, button):
#         """
#         """

#         self.matrix_ = co_occurrence_matrix(
#             x=self.data_,
#             column=self.column,
#             by=self.by,
#             top_by=self.top_by,
#             top_n=self.top_n,
#             limit_to=self.limit_to_,
#             exclude=self.exclude_,
#             sort_by=self.sort_by,
#             ascending=self.ascending,
#         )

#         self.output_.clear_output()
#         with self.output_:

#             if self.matrix_ is None:
#                 display(widgets.HTML("Different columns must be selected!"))
#                 return

#             if self.view == "Matrix":
#                 display(
#                     self.matrix_.style.background_gradient(cmap=self.cmap, axis=None)
#                 )

#             if self.view == "Heatmap":
#                 display(
#                     plt.heatmap(
#                         self.matrix_, cmap=self.cmap, figsize=(self.width, self.height)
#                     )
#                 )

#             if self.view == "Bubble plot":
#                 display(
#                     plt.bubble(
#                         self.matrix_,
#                         axis=0,
#                         cmap=self.cmap,
#                         figsize=(self.width, self.height),
#                     )
#                 )

#             if self.view == "Slope chart":
#                 display(
#                     slope_chart(
#                         self.matrix_,
#                         figsize=(self.width, self.height),
#                         cmap_column=self.cmap,
#                         cmap_by=self.cmap_by,
#                     )
#                 )

#             if self.view == "Network":
#                 display(
#                     co_occurrence_map(
#                         matrix=self.matrix_,
#                         layout=self.layout,
#                         cmap_column=self.cmap,
#                         cmap_by=self.cmap_by,
#                         figsize=(self.width, self.height),
#                     )
#                 )

#             # if self.view == "Table":

#             #     result = self.matrix_.stack().to_frame().reset_index()
#             #     result.columns = [self.by, self.column, "Values"]

#             #     result = result[result["Values"] != 0]
#             #     result = result.sort_values(["Values"])
#             #     result = result.reset_index(drop=True)

#             #     if self.sort_by == "Alphabetic":
#             #         result = result.sort_values(
#             #             [by, column, "Values"], ascending=self.ascending
#             #         )

#             #     if self.sort_by == "Num Documents":
#             #         result["ND-column"] = result[column].map(
#             #             lambda w: w.split(" ")[-1].split(":")[0]
#             #         )
#             #         result["ND-by"] = result[by].map(
#             #             lambda w: w.split(" ")[-1].split(":")[0]
#             #         )
#             #         result = result.sort_values(
#             #             ["ND-by", "ND-column", "Values"], ascending=self.ascending
#             #         )
#             #         result.pop("ND-column")
#             #         result.pop("ND-by")

#             #     display(result)


# ###
# ###
# ###
# ###


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
#             options=[
#                 "Matrix",
#                 "Heatmap",
#                 "Bubble plot",
#                 "Network",
#                 "Slope chart",
#                 "Table",
#             ],
#         ),
#         gui.dropdown(desc="Column:", options=[z for z in COLUMNS if z in x.columns],),
#         gui.dropdown(desc="By:", options=[z for z in COLUMNS if z in x.columns],),
#         gui.dropdown(
#             desc="Top by:", options=["Values", "Num Documents", "Times Cited"],
#         ),
#         gui.top_n(),
#         gui.dropdown(
#             desc="Sort by:", options=["Alphabetic", "Num Documents", "Times Cited",],
#         ),
#         gui.ascending(),
#         gui.cmap(arg="cmap_column", desc="Colormap Col:"),
#         gui.cmap(arg="cmap_by", desc="Colormap By:"),
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
#         # Logic
#         #
#         view = kwargs["view"]
#         column = kwargs["column"]
#         by = kwargs["by"]
#         cmap_column = kwargs["cmap_column"]
#         cmap_by = kwargs["cmap_by"]
#         top_by = kwargs["top_by"]
#         top_n = int(kwargs["top_n"])
#         sort_by = kwargs["sort_by"]
#         ascending = kwargs["ascending"]
#         layout = kwargs["layout"]
#         width = int(kwargs["width"])
#         height = int(kwargs["height"])

#         left_panel[2]["widget"].options = [
#             z for z in COLUMNS if z in x.columns and z != column
#         ]
#         by = left_panel[2]["widget"].value

#         if view in ["Matrix", "Table"]:
#             left_panel[-4]["widget"].disabled = True
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = True
#             left_panel[-1]["widget"].disabled = True

#         if view == "Heatmap":
#             left_panel[-4]["widget"].disabled = True
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False

#         if view == "Bubble plot":
#             left_panel[-4]["widget"].disabled = True
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False

#         if view == "Network":
#             left_panel[-4]["widget"].disabled = False
#             left_panel[-3]["widget"].disabled = False
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False

#         if view == "Slope chart":
#             left_panel[-4]["widget"].disabled = False
#             left_panel[-3]["widget"].disabled = True
#             left_panel[-2]["widget"].disabled = False
#             left_panel[-1]["widget"].disabled = False

#         matrix = co_occurrence_matrix(
#             x,
#             column=column,
#             by=by,
#             top_by=top_by,
#             top_n=top_n,
#             sort_by=sort_by,
#             ascending=ascending,
#             limit_to=limit_to,
#             exclude=exclude,
#         )

#         output.clear_output()
#         with output:
#             if view == "Matrix":
#                 display(matrix.style.background_gradient(cmap=cmap_column, axis=None))

#             if view == "Heatmap":
#                 display(plt.heatmap(matrix, cmap=cmap_column, figsize=(width, height)))

#             if view == "Bubble plot":
#                 display(
#                     plt.bubble(
#                         matrix, axis=0, cmap=cmap_column, figsize=(width, height)
#                     )
#                 )

#             if view == "Slope chart":
#                 display(
#                     slope_chart(
#                         matrix,
#                         figsize=(width, height),
#                         cmap_column=cmap_column,
#                         cmap_by=cmap_by,
#                     )
#                 )

#             if view == "Network":
#                 display(
#                     co_occurrence_map(
#                         matrix=matrix,
#                         layout=layout,
#                         cmap_column=cmap_column,
#                         cmap_by=cmap_by,
#                         figsize=(width, height),
#                     )
#                 )

#             if view == "Table":
#                 result = matrix.stack().to_frame().reset_index()
#                 result.columns = [by, column, "Values"]

#                 result = result[result["Values"] != 0]
#                 result = result.sort_values(["Values"])
#                 result = result.reset_index(drop=True)

#                 if sort_by == "Alphabetic":
#                     result = result.sort_values(
#                         [by, column, "Values"], ascending=ascending
#                     )

#                 if sort_by == "Num Documents":
#                     result["ND-column"] = result[column].map(
#                         lambda w: w.split(" ")[-1].split(":")[0]
#                     )
#                     result["ND-by"] = result[by].map(
#                         lambda w: w.split(" ")[-1].split(":")[0]
#                     )
#                     result = result.sort_values(
#                         ["ND-by", "ND-column", "Values"], ascending=ascending
#                     )
#                     result.pop("ND-column")
#                     result.pop("ND-by")

#                 display(result)

#         return

#     ###
#     output = widgets.Output()
#     return gui.TABapp(left_panel=left_panel, server=server, output=output)


###############################################################################
##
##  CALCULATIONS
##
###############################################################################


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
    top_terms = by_term.analytics(x, column)

    if isinstance(top_by, str):
        top_by = top_by.replace(" ", "_")
        top_by = {"Num_Documents": 0, "Times_Cited": 1,}[top_by]

    if top_by == 0:
        top_terms = top_terms.sort_values(
            ["Num_Documents", "Times_Cited"], ascending=[False, False],
        )

    if top_by == 1:
        top_terms = top_terms.sort_values(
            ["Times_Cited", "Num_Documents"], ascending=[False, False, True],
        )

    #  top_terms = top_terms[column]

    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        top_terms = top_terms[top_terms.index.map(lambda w: w in limit_to)]

    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        top_terms = top_terms[top_terms.index.map(lambda w: w not in exclude)]

    if top_n is not None:
        top_terms = top_terms.head(top_n)

    if isinstance(sort_by, str):
        sort_by = sort_by.replace(" ", "_")
        sort_by = {"Alphabetic": 0, "Num_Documents": 1, "Times_Cited": 2,}[sort_by]

    if isinstance(ascending, str):
        ascending = {"True": True, "False": False}[ascending]

    if sort_by == 0:
        top_terms = top_terms.sort_index(ascending=ascending)

    if sort_by == 1:
        top_terms = top_terms.sort_values(
            ["Num_Documents", "Times_Cited"], ascending=ascending
        )

    if sort_by == 2:
        top_terms = top_terms.sort_values(
            ["Times_Cited", "Num_Documents"], ascending=ascending
        )

    top_terms = top_terms.index.tolist()

    if axis == 0 or axis == 2:
        matrix = matrix.loc[[t for t in top_terms if t in matrix.index], :]

    if axis == 1 or axis == 2:
        matrix = matrix.loc[:, [t for t in top_terms if t in matrix.columns]]

    return matrix


def _get_fmt(summ):
    n_Num_Documents = int(np.log10(summ["Num_Documents"].max())) + 1
    n_Times_Cited = int(np.log10(summ["Times_Cited"].max())) + 1
    return "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"


#
#
#
#
#


# def occurrence_chord_diagram(
#     matrix, summary, cmap="Greys", figsize=(17, 12),
# ):
#     x = matrix.copy()
#     # ---------------------------------------------------
#     #
#     # Node sizes
#     #
#     #
#     # Data preparation
#     #
#     terms = matrix.columns.tolist()
#     terms = [w[: w.find("[")].strip() if "[" in w else w for w in terms]
#     terms = [w.strip() for w in terms]

#     num_documents = {k: v for k, v in zip(summary.index, summary["Num_Documents"])}
#     times_cited = {k: v for k, v in zip(summary.index, summary["Times_Cited"])}

#     #
#     # Node sizes
#     #
#     node_sizes = [num_documents[t] for t in terms]
#     min_size = min(node_sizes)
#     max_size = max(node_sizes)
#     if min_size == max_size:
#         node_sizes = [500 for t in terms]
#     else:
#         node_sizes = [
#             600 + int(2500 * (t - min_size) / (max_size - min_size)) for t in node_sizes
#         ]

#     #
#     # Node colors
#     #
#     cmap = pyplot.cm.get_cmap(cmap)
#     node_colors = [times_cited[t] for t in terms]
#     min_color = min(node_colors)
#     max_color = max(node_colors)
#     if min_color == max_color:
#         node_colors = [cmap(0.8) for t in terms]
#     else:
#         node_colors = [
#             cmap(0.2 + 0.75 * (t - min_color) / (max_color - min_color))
#             for t in node_colors
#         ]

#     #
#     # ---------------------------------------------------

#     cd = ChordDiagram()
#     for idx, term in enumerate(x.columns):
#         cd.add_node(term, color=node_colors[idx], s=node_sizes[idx])

#     max_value = x.max().max()
#     for idx_col in range(len(x.columns) - 1):
#         for idx_row in range(idx_col + 1, len(x.columns)):

#             node_a = x.index[idx_row]
#             node_b = x.columns[idx_col]
#             value = x[node_b][node_a]

#             cd.add_edge(
#                 node_a,
#                 node_b,
#                 linestyle="-",
#                 linewidth=4 * value / max_value,
#                 color="k",
#             )

#     return cd.plot(figsize=figsize)


#
# Associaction
#


# def associations_map(
#     matrix,
#     layout="Kamada Kawai",
#     cmap_column="Greys",
#     cmap_by="Greys",
#     figsize=(17, 12),
# ):
#     """Computes the occurrence map directly using networkx.
#     """

#     cmap_column = pyplot.cm.get_cmap(cmap_column)
#     cmap_by = pyplot.cm.get_cmap(cmap_by)

#     #
#     # Sizes
#     #
#     index_node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.index]
#     column_node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.columns]

#     min_size = min(index_node_sizes + column_node_sizes)
#     max_size = max(index_node_sizes + column_node_sizes)

#     index_node_sizes = [
#         150 + 2000 * (t - min_size) / (max_size - min_size) for t in index_node_sizes
#     ]
#     column_node_sizes = [
#         150 + 2000 * (t - min_size) / (max_size - min_size) for t in column_node_sizes
#     ]

#     #
#     # Colors
#     #
#     index_node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.index]
#     column_node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.columns]

#     min_color = min(index_node_colors + column_node_colors)
#     max_color = max(index_node_colors + column_node_colors)

#     index_node_colors = [
#         cmap_by(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
#         for t in index_node_colors
#     ]
#     column_node_colors = [
#         cmap_column(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
#         for t in column_node_colors
#     ]

#     terms = matrix.columns.tolist() + matrix.index.tolist()

#     #
#     # Draw the network
#     #
#     fig = pyplot.Figure(figsize=figsize)
#     ax = fig.subplots()

#     G = nx.Graph(ax=ax)
#     G.clear()

#     #
#     # network nodes
#     #
#     # G.add_nodes_from(terms)

#     #
#     # network edges
#     #
#     n = len(matrix.columns)
#     max_width = 0
#     for col in matrix.columns:
#         for row in matrix.index:
#             link = matrix.at[row, col]
#             if link > 0:
#                 G.add_edge(row, col, width=link)
#                 if max_width < link:
#                     max_width = link

#     #
#     # Layout
#     #
#     pos = {
#         "Circular": nx.circular_layout,
#         "Kamada Kawai": nx.kamada_kawai_layout,
#         "Planar": nx.planar_layout,
#         "Random": nx.random_layout,
#         "Spectral": nx.spectral_layout,
#         "Spring": nx.spring_layout,
#         "Shell": nx.shell_layout,
#     }[layout](G)

#     # draw_dict = {
#     #     "Circular": nx.draw_circular,
#     #     "Kamada Kawai": nx.draw_kamada_kawai,
#     #     "Planar": nx.draw_planar,
#     #     "Random": nx.draw_random,
#     #     "Spectral": nx.draw_spectral,
#     #     "Spring": nx.draw_spring,
#     #     "Shell": nx.draw_shell,
#     # }
#     # draw = draw_dict[layout]

#     for e in G.edges.data():
#         a, b, width = e
#         edge = [(a, b)]
#         width = 0.2 + 4.0 * width["width"] / max_width
#         nx.draw_networkx_edges(
#             G,
#             pos=pos,
#             ax=ax,
#             edgelist=edge,
#             width=width,
#             edge_color="k",
#             with_labels=False,
#             node_size=1,
#         )

#     #
#     # Draw column nodes
#     #
#     nx.draw_networkx_nodes(
#         G,
#         pos,
#         ax=ax,
#         edge_color="k",
#         nodelist=matrix.columns.tolist(),
#         node_size=column_node_sizes,
#         node_color=column_node_colors,
#         node_shape="o",
#         edgecolors="k",
#         linewidths=1,
#     )

#     #
#     # Draw index nodes
#     #
#     nx.draw_networkx_nodes(
#         G,
#         pos,
#         ax=ax,
#         edge_color="k",
#         nodelist=matrix.index.tolist(),
#         node_size=index_node_sizes,
#         node_color=index_node_colors,
#         node_shape="o",
#         edgecolors="k",
#         linewidths=1,
#     )

#     node_sizes = column_node_sizes + index_node_sizes

#     common.ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)

#     fig.tight_layout()
#     common.ax_expand_limits(ax)
#     ax.set_aspect("equal")
#     common.set_ax_splines_invisible(ax)

#     ax.axis("off")

#     return fig


#
# Association Map
#
# def __TAB1__(x, limit_to, exclude):
#     # -------------------------------------------------------------------------
#     #
#     # UI
#     #
#     # -------------------------------------------------------------------------
#     COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
#     #
#     left_panel = [
#         # 0
#         gui.dropdown(desc="Column:", options=[z for z in COLUMNS if z in x.columns],),
#         gui.dropdown(desc="By:", options=[z for z in COLUMNS if z in x.columns],),
#         gui.dropdown(
#             desc="Top by:", options=["Values", "Num Documents", "Times Cited",],
#         ),
#         gui.top_n(),
#         # 4
#         {
#             "arg": "cmap_column",
#             "desc": "Cmap col:",
#             "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
#         },
#         # 5
#         {
#             "arg": "cmap_by",
#             "desc": "Cmap by:",
#             "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
#         },
#         # 6
#         gui.nx_layout(),
#         gui.fig_width(),
#         gui.fig_height(),
#         # 9
#         {
#             "arg": "selected",
#             "desc": "Seleted Cols:",
#             "widget": widgets.widgets.SelectMultiple(
#                 options=[], layout=Layout(width="95%", height="150px"),
#             ),
#         },
#     ]
#     # -------------------------------------------------------------------------
#     #
#     # Logic
#     #
#     # -------------------------------------------------------------------------
#     def server(**kwargs):
#         #
#         # Logic
#         #
#         column = kwargs["column"]
#         by = kwargs["by"]
#         cmap_column = kwargs["cmap_column"]
#         cmap_by = kwargs["cmap_by"]
#         top_by = kwargs["top_by"]
#         top_n = int(kwargs["top_n"])
#         layout = kwargs["layout"]
#         width = int(kwargs["width"])
#         height = int(kwargs["height"])
#         selected = kwargs["selected"]

#         left_panel[1]["widget"].options = [
#             z for z in COLUMNS if z in x.columns and z != column
#         ]
#         by = left_panel[1]["widget"].value

#         matrix = co_occurrence_matrix(
#             x,
#             column=column,
#             by=by,
#             top_by=top_by,
#             top_n=top_n,
#             limit_to=limit_to,
#             exclude=exclude,
#         )

#         left_panel[-1]["widget"].options = sorted(matrix.columns)

#         if len(selected) == 0:
#             output.clear_output()
#             with output:
#                 display("No columns selected to analyze")
#                 return

#         Y = matrix[[t for t in matrix.columns if t in selected]]
#         S = Y.sum(axis=1)
#         S = S[S > 0]
#         Y = Y.loc[S.index, :]
#         if len(Y) == 0:
#             output.clear_output()
#             with output:
#                 display("There are not associations to show")
#                 return

#         output.clear_output()
#         with output:
#             display(
#                 associations_map(
#                     Y,
#                     layout=layout,
#                     cmap_column=cmap_column,
#                     cmap_by=cmap_by,
#                     figsize=(width, height),
#                 )
#             )
#         #
#         return

#     ###
#     output = widgets.Output()
#     return gui.TABapp(left_panel=left_panel, server=server, output=output)

