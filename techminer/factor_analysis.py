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
from document_term import TF_matrix
from explode import __explode
from keywords import Keywords
from plots import COLORMAPS
import common as cmn
import gui as gui


from matplotlib.lines import Line2D
from dashboard import DASH

###################################################################################################
##
##  CALCULATIONS
##
###################################################################################################


class FactorAnalysis:
    def __init__(self, data, limit_to, exclude):
        self._data = data
        self._limit_to = limit_to
        self._exclude = exclude

    def calculate(self, column, n_components, random_state):
        #
        X = self._data.copy()

        #
        # 1.-- Term-frequency matrix
        #
        TF_matrix_ = TF_matrix(X, column)

        #
        # 2.-- PCA (VantagePoint pag 170)
        #
        pca = PCA(n_components=n_components, random_state=random_state)
        R = np.transpose(pca.fit(X=TF_matrix_.values).components_)
        R = pd.DataFrame(
            R,
            columns=["F" + str(i) for i in range(n_components)],
            index=TF_matrix_.columns,
        )

        #
        # 3.-- limit to/exclude terms
        #
        R = cmn.limit_to_exclude(
            data=R,
            axis=0,
            column=column,
            limit_to=self._limit_to,
            exclude=self._exclude,
        )
        R = cmn.add_counters_to_axis(X=R, axis=0, data=X, column=column)

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

    def map(self, top_by, top_n, cmap, layout, iterations, figsize):

        X = self.factors_.copy()

        if top_by == "Values":

            m = X.copy()
            m = m.applymap(abs)
            m = m.max(axis=1)
            m = m.sort_values(ascending=False)
            m = m.head(top_n)
            m = m.index
            X = X.loc[m, :]

        else:

            X = cmn.sort_by_axis(data=X, sort_by=top_by, ascending=False, axis=0)
            X = X.head(top_n)

        ## Networkx
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()
        G = nx.Graph(ax=ax)
        G.clear()

        ## Data preparation
        terms = X.index.tolist()
        node_sizes = cmn.counters_to_node_sizes(x=terms)
        node_colors = cmn.counters_to_node_colors(
            x=terms, cmap=pyplot.cm.get_cmap(cmap)
        )

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

    def sort(self, top_by, top_n, sort_by, ascending):

        R = self.factors_.copy()

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

        if sort_by in ["F{}".format(i) for i in range(len(R.columns))]:
            R = R.sort_values(sort_by, ascending=ascending)

        if len(R) == 0:
            return None

        return R

    def update(
        self,
        output,
        top_by,
        top_n,
        sort_by,
        ascending,
        cmap,
        figsize,
        layout,
        iterations,
    ):

        if output == "Factors":
            return self.sort(
                top_by=top_by, top_n=top_n, sort_by=sort_by, ascending=ascending
            )

        if output == "Variances":
            return self.variances_

        if output == "Map":
            return self.map(
                top_by=top_by,
                top_n=top_n,
                cmap=cmap,
                layout=layout,
                iterations=iterations,
                figsize=figsize,
            )


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()


# def app(data, limit_to=None, exclude=None, tab=None):
#     return gui.APP(
#         app_title="Factor Analysis",
#         tab_titles=["Factor Analysis"],
#         tab_widgets=[TABapp0(data, limit_to=limit_to, exclude=exclude).run(),],
#         tab=tab,
#     )


###################################################################################################
##
##  DASHBOARD --- Factor Analysis
##
###################################################################################################

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


class DASHapp(DASH):
    def __init__(self, data, limit_to, exclude):
        #
        super(DASH, self).__init__()
        #
        self._obj = FactorAnalysis(data=data, limit_to=limit_to, exclude=exclude)
        #
        self.data_ = data
        self.limit_to_ = limit_to
        self.exclude_ = exclude
        self.app_title_ = "Factor Analysis"
        self.calculate_menu_options_ = None
        self.calculate_panel_ = [
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.n_components(),
            gui.random_state(),
        ]
        self.update_menu_options_ = ["Factors", "Variances", "Map"]
        self.update_panel_ = [
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
        #
        self.calculate_has_run = False
        #
        super().create_grid()
        #

    def run(self):
        return self.grid_

    def gui(self, **kwargs):
        super().gui(**kwargs)

        self.update_panel_[2]["widget"].disabled = (
            not self.update_menu_.value == "Factors"
        )
        self.update_panel_[3]["widget"].disabled = (
            not self.update_menu_.value == "Factors"
        )
        self.update_panel_[4]["widget"].disabled = (
            True if self.update_menu_.value == "Variances" else False
        )
        self.update_panel_[5]["widget"].disabled = (
            True if self.update_menu_.value != "Map" else False
        )
        self.update_panel_[6]["widget"].disabled = (
            True if self.update_menu_.value != "Map" else False
        )
        self.update_panel_[7]["widget"].disabled = (
            True if self.update_menu_.value != "Map" else False
        )
        self.update_panel_[8]["widget"].disabled = (
            True if self.update_menu_.value != "Map" else False
        )

        self.update_panel_[2]["widget"].options = [
            "Alphabetic",
            "Num Documents",
            "Times Cited",
        ] + ["F{}".format(i) for i in range(self.n_components)]

    def calculate(self, button):

        self.calculate_has_run = True

        self.output_.clear_output()
        with self.output_:
            display(gui.processing())
        self._obj.calculate(
            column=self.column,
            n_components=self.n_components,
            random_state=int(self.random_state),
        )
        self.update(button=None)

    def update(self, button):

        if self.calculate_has_run is False:
            self.calculate(button=None)
            return

        self.output_.clear_output()
        with self.output_:
            display(gui.processing())

        output = self._obj.update(
            output=self.update_menu,
            top_by=self.top_by,
            top_n=self.top_n,
            sort_by=self.sort_by,
            ascending=self.ascending,
            cmap=self.cmap,
            figsize=(self.width, self.height),
            layout=self.layout,
            iterations=self.nx_max_iter,
        )

        if self.update_menu == "Factors":
            cmap = pyplot.cm.get_cmap(self.cmap)
            output = output.style.format("{:.3f}").applymap(
                lambda w: "background-color: {}".format(colors.rgb2hex(cmap(abs(w))))
            )

        self.output_.clear_output()
        with self.output_:
            display(output)
