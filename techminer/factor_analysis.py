"""
Factor analysis
==================================================================================================



"""
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
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
from techminer.dashboard import DASH

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude):
        #
        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude
        #
        self.ascending = None
        self.cmap = None
        self.column = None
        self.height = None
        self.iterations = None
        self.layout = None
        self.n_components = None
        self.random_state = None
        self.sort_by = None
        self.top_by = None
        self.top_n = None
        self.width = None

    def fit(self):
        #
        X = self.data.copy()

        #
        # 1.-- Term-frequency matrix
        #
        TF_matrix_ = TF_matrix(X, self.column)

        #
        # 2.-- PCA (VantagePoint pag 170)
        #
        pca = PCA(n_components=self.n_components, random_state=int(self.random_state))
        R = np.transpose(pca.fit(X=TF_matrix_.values).components_)
        R = pd.DataFrame(
            R,
            columns=["F" + str(i) for i in range(self.n_components)],
            index=TF_matrix_.columns,
        )

        #
        # 3.-- limit to/exclude terms
        #
        R = cmn.limit_to_exclude(
            data=R,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        R = cmn.add_counters_to_axis(X=R, axis=0, data=X, column=self.column)

        #
        # 4.-- Top 50 values
        #
        m = R.copy()
        set0 = m.applymap(abs).max(axis=1).sort_values(ascending=False).head(50).index
        set1 = (
            cmn.sort_by_axis(data=R, sort_by="Num Documents", ascending=False, axis=0)
            .head(50)
            .index
        )
        set2 = (
            cmn.sort_by_axis(data=R, sort_by="Times Cited", ascending=False, axis=0)
            .head(50)
            .index
        )
        terms = set0 | set1 | set2
        R = R.loc[terms, :]

        self.factors_ = R
        self.variances_ = pd.DataFrame(
            pca.explained_variance_, columns=["Explained variance"]
        )
        self.variances_["Explained ratio"] = pca.explained_variance_ratio_
        self.variances_["Singular values"] = pca.singular_values_

    def sort(self):

        R = self.factors_.copy()

        if self.top_by == "Values":

            m = R.copy()
            m = m.applymap(abs)
            m = m.max(axis=1)
            m = m.sort_values(ascending=False)
            m = m.head(self.top_n)
            m = m.index
            R = R.loc[m, :]

        else:

            R = cmn.sort_by_axis(data=R, sort_by=self.top_by, ascending=False, axis=0)
            R = R.head(self.top_n)

        if self.sort_by in ["Alphabetic", "Num Documents", "Times Cited"]:
            R = cmn.sort_by_axis(
                data=R, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )

        if self.sort_by in ["F{}".format(i) for i in range(len(R.columns))]:
            R = R.sort_values(self.sort_by, ascending=self.ascending)

        if len(R) == 0:
            return None

        return R

    def factors(self):
        self.fit()
        output = self.sort()
        if self.cmap is None:
            return output
        else:
            return output.style.background_gradient(cmap=self.cmap)

    def variances(self):
        self.fit()
        return self.variances_

    def map(self):
        self.fit()
        X = self.factors_.copy()

        if self.top_by == "Values":

            m = X.copy()
            m = m.applymap(abs)
            m = m.max(axis=1)
            m = m.sort_values(ascending=False)
            m = m.head(self.top_n)
            m = m.index
            X = X.loc[m, :]

        else:

            X = cmn.sort_by_axis(data=X, sort_by=self.top_by, ascending=False, axis=0)
            X = X.head(self.top_n)

        ## Networkx
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        G = nx.Graph(ax=ax)
        G.clear()

        ## Data preparation
        terms = X.index.tolist()
        node_sizes = cmn.counters_to_node_sizes(x=terms)
        node_colors = cmn.counters_to_node_colors(
            x=terms, cmap=pyplot.cm.get_cmap(self.cmap)
        )

        ## Add nodes
        G.add_nodes_from(terms)

        ## node positions
        if self.layout == "Spring":
            pos = nx.spring_layout(G, iterations=self.iterations)
        else:
            pos = {
                "Circular": nx.circular_layout,
                "Kamada Kawai": nx.kamada_kawai_layout,
                "Planar": nx.planar_layout,
                "Random": nx.random_layout,
                "Spectral": nx.spectral_layout,
                "Spring": nx.spring_layout,
                "Shell": nx.shell_layout,
            }[self.layout](G)

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
                        n_edges_0 += (
                            1 if m.link_[idx] > 0 and m.link_[idx] < 0.25 else 0
                        )

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
        cmn.ax_text_node_labels(
            ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes
        )

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


###############################################################################
##
##  DASHBOARD
##
###############################################################################

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


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None):
        """Dashboard app"""

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.data = data
        self.app_title = "Factor Analysis"
        self.menu_options = ["Factors", "Variances", "Map"]

        self.panel_widgets = [
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.n_components(),
            gui.random_state(),
            gui.dropdown(
                desc="Top by:", options=["Values", "Num Documents", "Times Cited"],
            ),
            gui.top_n(),
            gui.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "Factor",],
            ),
            gui.ascending(),
            gui.cmap(),
            gui.nx_layout(),
            gui.nx_iterations(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        self.panel_widgets[5]["widget"].disabled = not self.menu == "Factors"
        self.panel_widgets[6]["widget"].disabled = not self.menu == "Factors"
        self.panel_widgets[7]["widget"].disabled = (
            True if self.menu == "Variances" else False
        )
        self.panel_widgets[8]["widget"].disabled = True if self.menu != "Map" else False
        self.panel_widgets[9]["widget"].disabled = True if self.menu != "Map" else False
        self.panel_widgets[10]["widget"].disabled = (
            True if self.menu != "Map" else False
        )
        self.panel_widgets[11]["widget"].disabled = (
            True if self.menu != "Map" else False
        )
        self.panel_widgets[9]["widget"].disabled = (
            True if self.panel_widgets[8]["widget"].value != "Spring" else False
        )

        self.panel_widgets[5]["widget"].options = [
            "Alphabetic",
            "Num Documents",
            "Times Cited",
        ] + ["F{}".format(i) for i in range(self.n_components)]


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()

