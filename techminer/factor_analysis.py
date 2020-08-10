"""
Factor analysis
==================================================================================================



"""
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.cluster import (
    AgglomerativeClustering,
    AffinityPropagation,
    Birch,
    DBSCAN,
    FeatureAgglomeration,
    KMeans,
    MeanShift,
)
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
import matplotlib.pyplot as pyplot

import techminer.common as cmn
import techminer.dashboard as dash
from techminer.dashboard import DASH
from techminer.document_term import TF_matrix, TFIDF_matrix
from techminer.graph import network_normalization


from matplotlib import patches


###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def fit(self):
        #
        X = self.data.copy()

        #
        # 1.-- TF matrix
        #
        TF_matrix_ = TF_matrix(
            data=X, column=self.column, scheme=None, min_occurrence=self.min_occurrence,
        )

        #
        # 2.-- Limit to / Exclude
        #
        TF_matrix_ = cmn.limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        #
        # 4.-- Add counters to axes
        #
        TF_matrix_ = cmn.add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 3.-- Select top terms
        #
        TF_matrix_ = cmn.sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        #
        # 3.-- Co-occurrence matrix and normalization
        #

        M = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        M = pd.DataFrame(M, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        M = network_normalization(M, normalization=self.normalization)

        #
        # 4.-- Dissimilarity matrix
        #
        if self.normalization == "None":
            M = M.max().max() - M
        else:
            M = 1 - M
            for i in M.columns.tolist():
                M.at[i, i] = 0.0
        #
        # 5.-- Factor decomposition
        #
        model = {
            "Factor Analysis": FactorAnalysis,
            "PCA": PCA,
            "Fast ICA": FastICA,
            "SVD": TruncatedSVD,
            "MDS": MDS,
        }[self.method]

        if self.method == "MDS":
            model = model(
                n_components=self.n_components,
                random_state=int(self.random_state),
                dissimilarity="precomputed",
            )
        else:

            model = model(
                n_components=self.n_components, random_state=int(self.random_state)
            )

        R = model.fit_transform(X=M.values)
        R = pd.DataFrame(
            R,
            columns=["Dim-{:>02d}".format(i) for i in range(self.n_components)],
            index=M.columns,
        )

        #
        # 6.-- Clustering
        #
        ## from bibliometrix: clustering of two first axes
        W = R.loc[:, R.columns[[0, 1]]]
        ##
        if self.clustering_method == "Affinity Propagation":
            labels = AffinityPropagation(
                random_state=int(self.random_state)
            ).fit_predict(W)
            self.n_clusters = len(set(labels))

        if self.clustering_method == "Agglomerative Clustering":
            labels = AgglomerativeClustering(
                n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage
            ).fit_predict(W)

        if self.clustering_method == "Birch":
            labels = Birch(n_clusters=self.n_clusters).fit_predict(W)

        if self.clustering_method == "DBSCAN":
            labels = DBSCAN().fit_predict(W)
            self.n_clusters = len(set(labels))

        #  if self.clustering_method == "Feature Agglomeration":
        #      m = FeatureAgglomeration(
        #          n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage
        #      ).fit(1 - X)
        #      labels = ???

        if self.clustering_method == "KMeans":
            labels = KMeans(
                n_clusters=self.n_clusters, random_state=int(self.random_state)
            ).fit_predict(W)

        if self.clustering_method == "Mean Shift":
            labels = MeanShift().fit_predct(W)
            self.n_clusters = len(set(labels))

        R["Cluster"] = labels
        self.coordinates_ = R

        #
        # 7.-- Cluster centers
        #
        self.centers_ = R.groupby("Cluster").mean()

        #
        # 8.-- Communities
        #
        communities = pd.DataFrame(
            pd.NA, columns=range(self.n_clusters), index=range(self.top_n)
        )
        for i_cluster in range(self.n_clusters):
            X = R[R.Cluster == i_cluster]
            X = cmn.sort_axis(
                data=X,
                num_documents=(self.top_by == "Num Documents"),
                axis=0,
                ascending=False,
            )
            community = X.index
            community = community.tolist()[0 : self.top_n]
            communities.at[0 : len(community) - 1, i_cluster] = community
        communities.columns = ["Cluster {}".format(i) for i in range(self.n_clusters)]
        ## Delete empty rows
        row_ids = []
        for row in communities.iterrows():
            if any([not pd.isna(a) for a in row[1]]):
                row_ids.append(row[0])
        communities = communities.loc[row_ids, :]
        communities = communities.applymap(lambda w: "" if pd.isna(w) else w)
        ##
        self.communities_ = communities

        #
        # 8.-- Cluster name
        #
        names = communities.loc[0, :].tolist()
        self.centers_["Name"] = names

        #
        # 10.-- Results
        #
        self.X_ = R

    def communities(self):
        self.fit()
        return self.communities_

    def mds_map(self):
        #
        def encircle(x, y, ax, **kw):
            p = np.c_[x, y]
            hull = ConvexHull(p, qhull_options="QJ")
            poly = pyplot.Polygon(p[hull.vertices, :], **kw)
            ax.add_patch(poly)

        #
        # 1.-- Creates the plot in memory
        #
        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()

        #
        # 2.-- Data clustering
        #
        self.fit()

        #
        # 3.-- Cluster colors (max 100 clusters)
        # º
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
            "cornflowerblue",
            "lightsalmon",
            "limegreen",
            "tomato",
            "mediumvioletred",
            "darkgoldenrod",
            "lightcoral",
            "silver",
            "darkkhaki",
            "skyblue",
            "dodgerblue",
            "orangered",
            "turquoise",
            "crimson",
            "violet",
            "goldenrod",
            "thistle",
            "grey",
            "yellowgreen",
            "lightcyan",
        ]

        colors += colors + colors

        #
        # 3.-- Plot the points of each cluster
        #
        R = self.coordinates_
        for i_cluster in range(self.n_clusters):
            X = R[R.Cluster == i_cluster]
            X = cmn.sort_axis(
                data=X,
                num_documents=(self.top_by == "Num Documents"),
                axis=0,
                ascending=False,
            )
            X.pop("Cluster")
            X = X.head(self.top_n)
            x = X[X.columns[self.x_axis]]
            y = X[X.columns[self.y_axis]]
            ax.scatter(
                x, y, marker="o", s=12, alpha=0.8, c=colors[i_cluster],
            )

            if len(X) > 2:
                encircle(x, y, ax=ax, ec="k", fc=colors[i_cluster], alpha=0.1)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            for x_, y_, t in zip(x.tolist(), y.tolist(), X.index):
                ax.text(
                    x=x_ + 0.01 * (xlim[1] - xlim[0]),
                    y=y_ - 0.01 * (ylim[1] - ylim[0]),
                    s=t,
                    fontsize=9,
                    color=colors[i_cluster],
                    horizontalalignment="left",
                    verticalalignment="top",
                )

        # ------------------------------------------------------------------------------------
        #  cmap = pyplot.cm.get_cmap("Greys")
        #  x = self.centers_["Dim-{:>02d}".format(self.x_axis)]
        #  y = self.centers_["Dim-{:>02d}".format(self.y_axis)]
        #  names = self.centers_["Name"]
        # for i_cluster in range(self.n_clusters):
        #     ax.text(
        #         x=x[i_cluster],
        #         y=y[i_cluster],
        #         s=names[i_cluster],
        #         fontsize=10,
        #         bbox=dict(
        #             facecolor="w",
        #             alpha=1.0,
        #             edgecolor="gray",
        #             boxstyle="round,pad=0.5",
        #         ),
        #         horizontalalignment="left",
        #         verticalalignment="top",
        #     )
        # ------------------------------------------------------------------------------------

        ax.axhline(
            y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
        )
        ax.axvline(
            x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
        )
        ax.axis("off")
        ax.set_aspect("equal")
        cmn.set_ax_splines_invisible(ax)
        ax.grid(axis="both", color="lightgray", linestyle="--", linewidth=0.5)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(
            x=xlim[1],
            y=0.01 * (ylim[1] - ylim[0]),
            s="Dim-0",
            fontsize=9,
            color="dimgray",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.text(
            x=0.01 * (xlim[1] - xlim[0]),
            y=ylim[1],
            s="Dim-1",
            fontsize=9,
            color="dimgray",
            horizontalalignment="left",
            verticalalignment="top",
        )

        fig.set_tight_layout(True)
        return fig

    def cluster_plot(self):
        ##
        self.fit()
        ##

        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        x = self.centers_["Dim-{:>02d}".format(self.x_axis)]
        y = self.centers_["Dim-{:>02d}".format(self.y_axis)]
        names = self.centers_["Name"]
        node_sizes = cmn.counters_to_node_sizes(names)
        #  node_colors = cmn.counters_to_node_colors(names, cmap)

        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
            "cornflowerblue",
            "lightsalmon",
            "limegreen",
            "tomato",
            "mediumvioletred",
            "darkgoldenrod",
            "lightcoral",
            "silver",
            "darkkhaki",
            "skyblue",
            "dodgerblue",
            "orangered",
            "turquoise",
            "crimson",
            "violet",
            "goldenrod",
            "thistle",
            "grey",
            "yellowgreen",
            "lightcyan",
        ]

        colors += colors + colors

        ax.scatter(
            x,
            y,
            marker="o",
            s=node_sizes,
            c=colors[: len(x)],
            #  c=node_colors,
            alpha=0.5,
            linewidths=2,
            #  edgecolors=node_colors),
        )

        pos = {term: (x[idx], y[idx]) for idx, term in enumerate(self.centers_.Name)}
        cmn.ax_text_node_labels(
            ax=ax, labels=self.centers_.Name, dict_pos=pos, node_sizes=node_sizes
        )

        cmn.ax_expand_limits(ax)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(
            x=xlim[1],
            y=0.01 * (ylim[1] - ylim[0]),
            s="Dim-0",
            fontsize=9,
            color="dimgray",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.text(
            x=0.01 * (xlim[1] - xlim[0]),
            y=ylim[1],
            s="Dim-1",
            fontsize=9,
            color="dimgray",
            horizontalalignment="left",
            verticalalignment="top",
        )

        cmn.set_ax_splines_invisible(ax)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.set_axis_off()
        fig.set_tight_layout(True)

        return fig


###############################################################################
##
##  DASHBOARD
##
###############################################################################

COLUMNS = sorted(
    [
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
)


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.n_components = 2
        self.x_axis = 0
        self.y_axis = 1

        self.app_title = "Factor Analysis"
        self.menu_options = [
            "Communities",
            "Cluster plot",
            "MDS map",
        ]
        #
        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.dropdown(
                desc="Normalization:",
                options=[
                    "Association",
                    "Jaccard",
                    "Dice",
                    "Salton",
                    "Equivalence",
                    "Inclusion",
                    "Cosine",
                    "Mutual Information",
                ],
            ),
            dash.separator(text="Decomposition"),
            dash.dropdown(
                desc="Method:",
                options=["MDS", "Factor Analysis", "PCA", "Fast ICA", "SVD"],
            ),
            ## dash.n_components(),
            dash.random_state(),
            dash.separator(text="Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.random_state(),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
            dash.top_n(),
            #  dash.x_axis(),
            #  dash.y_axis(),
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

        if self.clustering_method in ["Affinity Propagation"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_enabled("Random State:")

        if self.clustering_method in ["Agglomerative Clustering"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            self.set_disabled("Random State:")

        if self.clustering_method in ["Birch"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_disabled("Random State:")

        if self.clustering_method in ["DBSCAN"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_disabled("Random State:")

        if self.clustering_method in ["Feature Agglomeration"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            self.set_disabled("Random State:")

        if self.clustering_method in ["KMeans"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_disabled("Random State:")

        if self.clustering_method in ["Mean Shift"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_disabled("Random State:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()
