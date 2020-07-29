"""
Factor analysis
==================================================================================================



"""
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
import matplotlib.pyplot as pyplot

import techminer.common as cmn
import techminer.dashboard as dash
from techminer.dashboard import DASH
from techminer.document_term import TF_matrix, TFIDF_matrix

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


class DASHapp(DASH):
    def __init__(self, data, limit_to=None, exclude=None):
        """Dashboard app"""

        # Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

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
        #
        self.app_title = "Factor Analysis"
        self.menu_options = [
            "Memberships",
            "Cluster plot",
        ]
        #
        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Decomposition"),
            dash.dropdown(desc="Apply to:", options=["TF matrix", "TF*IDF matrix",],),
            dash.dropdown(
                desc="Method:",
                options=["Factor Analysis", "PCA", "Fast ICA", "SVD", "MDS"],
            ),
            dash.n_components(),
            dash.random_state(),
            dash.separator(text="Aglomerative Clustering"),
            dash.n_clusters(),
            dash.affinity(),
            dash.linkage(),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
            dash.top_n(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Memberships":
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu == "Cluster plot":
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        self.set_options(name="X-axis:", options=list(range(self.n_components)))
        self.set_options(name="Y-axis:", options=list(range(self.n_components)))

    def fit(self):
        #
        X = self.data.copy()

        #
        # 1.-- TF matrix
        #
        M = TF_matrix(
            data=X, column=self.column, scheme=None, min_occurrence=self.min_occurrence,
        )

        #
        # 2.-- Computtes TFIDF matrix and select max_term frequent terms
        #
        #      tf-idf = tf * (log(N / df) + 1)
        #
        if self.apply_to == "TF*IDF matrix":
            M = TFIDF_matrix(
                TF_matrix=M,
                norm=None,
                use_idf=False,
                smooth_idf=False,
                sublinear_tf=False,
                max_items=self.max_items,
            )

        #
        # 3.-- Add counters to axes
        #
        M = cmn.add_counters_to_axis(X=M, axis=1, data=self.data, column=self.column)

        #
        # 4.-- Factor decomposition
        #
        model = {
            "Factor Analysis": FactorAnalysis,
            "PCA": PCA,
            "Fast ICA": FastICA,
            "SVD": TruncatedSVD,
            "MDS": MDS,
        }[self.method](
            n_components=self.n_components, random_state=int(self.random_state)
        )

        R = np.transpose(model.fit(X=M.values).components_)
        R = pd.DataFrame(
            R,
            columns=["Dim-{:>02d}".format(i) for i in range(self.n_components)],
            index=M.columns,
        )

        #
        # 5.-- limit to/exclude terms
        #
        R = cmn.limit_to_exclude(
            data=R,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        # R = cmn.add_counters_to_axis(X=R, axis=0, data=X, column=self.column)

        #
        # 6.-- Clustering
        #
        clustering = AgglomerativeClustering(
            n_clusters=int(self.n_clusters),
            affinity=self.affinity,
            linkage=self.linkage,
        )
        clustering.fit(R)
        R["Cluster"] = clustering.labels_

        #
        # 7.-- Cluster centers
        #
        self.centers_ = R.groupby("Cluster").mean()

        #
        # 8.-- Cluster name
        #
        names = []
        for i_cluster in range(self.n_clusters):
            X = R[R.Cluster == i_cluster]
            X = cmn.sort_axis(
                data=X,
                num_documents=(self.top_by == "Num Documents"),
                axis=0,
                ascending=False,
            )
            names.append(X.index[0])
        self.centers_["Name"] = names

        #
        # 8.-- Results
        #
        self.X_ = R

    def memberships(self):
        ##
        self.fit()
        ##
        HTML = ""
        for i_cluster in range(self.n_clusters):
            X = self.X_[self.X_.Cluster == i_cluster]
            X = cmn.sort_axis(
                data=X,
                num_documents=(self.top_by == "Num Documents"),
                axis=0,
                ascending=False,
            )
            X = X.head(self.top_n)

            HTML += (
                "==================================================================<br>"
            )
            HTML += "Cluster: " + str(i_cluster) + "<br>"
            for t in X.index:
                HTML += "    {:>45s}".format(t) + "<br>"
            HTML += "<br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")

    def cluster_plot(self):
        ##
        self.fit()
        ##

        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        #  cmap=pyplot.cm.get_cmap(self.cmap)
        cmap = pyplot.cm.get_cmap("Greys")

        x = self.centers_["Dim-{:>02d}".format(self.x_axis)]
        y = self.centers_["Dim-{:>02d}".format(self.y_axis)]
        names = self.centers_["Name"]
        node_sizes = cmn.counters_to_node_sizes(names)
        node_colors = cmn.counters_to_node_colors(names, cmap)

        ax.scatter(
            x,
            y,
            marker="o",
            s=node_sizes,
            c=node_colors,
            alpha=0.7,
            linewidths=1,
            edgecolors="k",
        )

        pos = {term: (x[idx], y[idx]) for idx, term in enumerate(self.centers_.Name)}
        cmn.ax_text_node_labels(
            ax=ax, labels=self.centers_.Name, dict_pos=pos, node_sizes=node_sizes
        )

        cmn.ax_expand_limits(ax)
        cmn.set_ax_splines_invisible(ax)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.set_axis_off()
        fig.set_tight_layout(True)

        return fig


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()
