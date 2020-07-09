"""
Factor analysis
==================================================================================================



"""
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
import techminer.by_term as by_term
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from matplotlib import colors
from sklearn.decomposition import PCA
from techminer.document_term import TF_matrix
from techminer.explode import __explode
from techminer.keywords import Keywords
from techminer.plots import COLORMAPS
import techminer.common as cmn

import techminer.gui as gui
from matplotlib.lines import Line2D

###################################################################################################
##
##  CALCULATIONS
##
###################################################################################################


def factor_analysis(
    X,
    column,
    n_components=2,
    top_by=None,
    top_n=None,
    sort_by=None,
    ascending=True,
    figsize=(10, 10),
    layout="Kamada Kawai",
    cmap=None,
    limit_to=None,
    exclude=None,
):

    X = X.copy()
    TF_matrix_ = TF_matrix(X, column)
    terms = TF_matrix_.columns.tolist()
    if n_components is None:
        n_components = int(np.sqrt(len(set(terms))))
    pca = PCA(n_components=n_components)
    R = np.transpose(pca.fit(X=TF_matrix_.values).components_)
    R = pd.DataFrame(
        R, columns=["F" + str(i) for i in range(n_components)], index=terms
    )

    R = cmn.limit_to_exclude(
        data=R, axis=0, column=column, limit_to=limit_to, exclude=exclude,
    )
    R = cmn.add_counters_to_axis(X=R, axis=0, data=X, column=column)

    if top_by == "Values":

        m = R.copy()
        m = m.applymap(abs)
        m = m.max(axis=1)
        m = m.sort_values(ascending=False)
        m = m.head(top_n)
        m = m.index
        R = R.loc[m, :]

    else:

        R = cmn.sort_by_axis(data=R, sort_by=top_by, ascending=False, axis=0)
        R = R.head(top_n)

    if sort_by in ["Alphabetic", "Num Documents", "Times Cited"]:
        R = cmn.sort_by_axis(data=R, sort_by=sort_by, ascending=ascending, axis=0)

    if sort_by in ["F{}".format(i) for i in range(n_components)]:
        R = R.sort_values(sort_by, ascending=ascending)

    if len(R) == 0:
        return None

    return R


# -------------------------------------------------------------------------------------------------


def factor_map(X, layout="Kamada Kawai", iterations=50, cmap="Greys", figsize=(17, 12)):
    """
    """

    ## Networkx
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    ## Data preparation
    terms = X.index.tolist()
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

    for factor in X.columns:

        for k in [0, 1]:

            M = X[[factor]]

            if k == 1:
                M = M.applymap(lambda w: -w)

            M = M[M[factor] >= 0.25]

            if len(M) > 0:
                F = M[[factor]].values.T + M[[factor]].values
                F = F / 2
                F = pd.DataFrame(F, columns=M.index, index=M.index)
                m = F.stack().to_frame().reset_index()
                m = m[m.level_0 < m.level_1]
                m.columns = ["from_", "to_", "link_"]
                m = m.reset_index(drop=True)

                for idx in range(len(m)):

                    edge = [(m.from_[idx], m.to_[idx])]
                    key = (
                        0
                        if m.link_[idx] > 0.75
                        else (
                            1
                            if m.link_[idx] > 0.50
                            else (2 if m.link_[idx] > 0.25 else 3)
                        )
                    )

                    n_edges_75 += 1 if m.link_[idx] >= 0.75 else 0
                    n_edges_50 += (
                        1 if m.link_[idx] >= 0.50 and m.link_[idx] < 0.75 else 0
                    )
                    n_edges_25 += (
                        1 if m.link_[idx] >= 0.25 and m.link_[idx] < 0.50 else 0
                    )
                    n_edges_0 += 1 if m.link_[idx] > 0 and m.link_[idx] < 0.25 else 0

                    nx.draw_networkx_edges(
                        G,
                        pos=pos,
                        ax=ax,
                        node_size=1,
                        with_labels=False,
                        edgelist=edge,
                        **(d[key])
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
        app_title="Factor Analysis",
        tab_titles=["Factor Analysis"],
        tab_widgets=[TABapp0(data, limit_to=limit_to, exclude=exclude).run(),],
        tab=tab,
    )


###################################################################################################
##
##  TAB app 0 --- Factor Analysis
##
###################################################################################################


class TABapp0(gui.TABapp_):
    def __init__(self, data, limit_to, exclude):

        super(TABapp0, self).__init__()

        self.data_ = data
        self.limit_to_ = limit_to
        self.exclude_ = exclude

        COLUMNS = [
            "Authors",
            "Countries",
            "Institutions",
            "Author_Keywords",
            "Index_Keywords",
            "Abstract_words_CL",
            "Abstract_words",
            "Title_words_CL",
            "Title_words",
            "Affiliations",
            "Author_Keywords_CL",
            "Index_Keywords_CL",
        ]

        self.panel_ = [
            gui.dropdown(desc="View:", options=["Matrix", "Network"],),
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.n_components(),
            gui.cmap(),
            gui.dropdown(
                desc="Top by:", options=["Values", "Num Documents", "Times Cited"],
            ),
            gui.top_n(),
            gui.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "Factor",],
            ),
            gui.ascending(),
            gui.nx_layout(),
            gui.nx_max_iters(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def gui(self, **kwargs):

        super().gui(**kwargs)

        self.panel_[6]["widget"].disabled = self.view == "Network"
        self.panel_[7]["widget"].disabled = self.view == "Network"

        self.panel_[-3]["widget"].disabled = (
            True if self.layout != "Spring" or self.view != "Network" else False
        )

        self.panel_[-4]["widget"].disabled = False if self.view in ["Network"] else True
        self.panel_[-2]["widget"].disabled = False if self.view in ["Network"] else True
        self.panel_[-1]["widget"].disabled = False if self.view in ["Network"] else True

        self.panel_[6]["widget"].options = [
            "Alphabetic",
            "Num Documents",
            "Times Cited",
        ] + ["F{}".format(i) for i in range(self.n_components)]

    def update(self, button):
        """ 
        """

        self.output_.clear_output()
        with self.output_:
            display(gui.processing())

        self.matrix_ = factor_analysis(
            X=self.data_,
            column=self.column,
            n_components=self.n_components,
            top_by=self.top_by,
            top_n=self.top_n,
            layout=self.layout,
            sort_by=self.sort_by,
            ascending=self.ascending,
            limit_to=self.limit_to_,
            exclude=self.exclude_,
        )

        self.output_.clear_output()
        with self.output_:

            if self.matrix_ is None:
                display(widgets.HTML("Factor matrix empty!"))
                return

            if self.view == "Matrix":
                cmap = pyplot.cm.get_cmap(self.cmap)
                display(
                    self.matrix_.style.format("{:.3f}").applymap(
                        lambda w: "background-color: {}".format(
                            colors.rgb2hex(cmap(abs(w)))
                        )
                    )
                )

            if self.view == "Network":
                display(
                    factor_map(
                        X=self.matrix_,
                        layout=self.layout,
                        cmap=self.cmap,
                        iterations=self.nx_max_iter,
                        figsize=(self.width, self.height),
                    )
                )

            if self.view == "Network0":
                display(
                    factor_map_0(
                        matrix=self.matrix_,
                        layout=self.layout,
                        cmap=self.cmap,
                        figsize=(self.width, self.height),
                    )
                )


#
#
# APP
#
#


# def __TAB0__(data, limit_to, exclude):
#     # -------------------------------------------------------------------------
#     #
#     # UI
#     #
#     # -------------------------------------------------------------------------

#     left_panel = [
#         gui.dropdown(desc="View:", options=["Matrix", "Network", "Network0", "Table"],),
#         gui.dropdown(desc="Term:", options=[z for z in COLUMNS if z in data.columns],),
#         gui.n_components(),
#         gui.cmap(),
#         gui.dropdown(
#             desc="Top by:", options=["Values", "Num Documents", "Times Cited"],
#         ),
#         gui.top_n(),
#         gui.dropdown(
#             desc="Sort by:",
#             options=["Alphabetic", "Num Documents/Times Cited", "Factor",],
#         ),
#         gui.ascending(),
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
#         view = kwargs["view"]
#         term = kwargs["term"]
#         n_components = int(kwargs["n_components"])
#         cmap = kwargs["cmap"]
#         top_by = kwargs["top_by"]
#         top_n = int(kwargs["top_n"])
#         sort_by = kwargs["sort_by"]
#         ascending = kwargs["ascending"]
#         layout = kwargs["layout"]
#         width = int(kwargs["width"])
#         height = int(kwargs["height"])

#         if view == "Table":

#             left_panel[6]["widget"].options = [
#                 term,
#                 "Factor",
#                 "Values",
#             ]
#             sort_by = left_panel[6]["widget"].value

#         else:

#             left_panel[6]["widget"].options = [
#                 "Alphabetic",
#                 "Num Documents",
#                 "Times Cited",
#             ] + ["F{}".format(i) for i in range(n_components)]
#             sort_by = left_panel[6]["widget"].value

#         left_panel[5]["widget"].disabled = True if top_by == "Values" else False
#         left_panel[-3]["widget"].disabled = (
#             False if view in ["Network", "Network0"] else True
#         )
#         left_panel[-2]["widget"].disabled = (
#             False if view in ["Network", "Network0"] else True
#         )
#         left_panel[-1]["widget"].disabled = (
#             False if view in ["Network", "Network0"] else True
#         )

#         output.clear_output()
#         with output:
#             return display(
#                 factor_analysis(
#                     data=data,
#                     column=term,
#                     output=view,
#                     n_components=n_components,
#                     top_by=top_by,
#                     top_n=top_n,
#                     layout=layout,
#                     sort_by=sort_by,
#                     ascending=ascending,
#                     cmap=cmap,
#                     figsize=(width, height),
#                     limit_to=limit_to,
#                     exclude=exclude,
#                 )
#             )

#     ###
#     output = widgets.Output()
#     return gui.TABapp(left_panel=left_panel, server=server, output=output)


# ###############################################################################
# ##
# ##  APP
# ##
# ###############################################################################


# def app(data, limit_to=None, exclude=None, tab=None):
#     return gui.APP(
#         app_title="Factor Analysis",
#         tab_titles=["Factor Analisis"],
#         tab_widgets=[__TAB0__(data, limit_to=limit_to, exclude=exclude),],
#         tab=tab,
#     )


#
#
#

#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
