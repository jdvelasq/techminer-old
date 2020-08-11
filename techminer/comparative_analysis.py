import json
import ipywidgets as widgets
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    AffinityPropagation,
    Birch,
    DBSCAN,
    FeatureAgglomeration,
    KMeans,
    MeanShift,
)

import matplotlib.pyplot as pyplot

import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.correspondence import CA
from techminer.dashboard import DASH
from techminer.diagram_plot import diagram_plot
from techminer.document_term import TF_matrix, TFIDF_matrix
from techminer.params import EXCLUDE_COLS


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

    def apply(self):

        ##
        ## Comparative analysis
        ##   from https://tlab.it/en/allegati/help_en_online/mcluster.htm
        ##

        #
        # 1.-- Computes TF matrix for terms in min_occurrence
        #
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occurrence,
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
        # 3.-- Computtes TFIDF matrix and select max_term frequent terms
        #
        #      tf-idf = tf * (log(N / df) + 1)
        #
        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=None,
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
            max_items=self.max_items,
        )

        #
        # 4.-- Adds counter to axies
        #
        TFIDF_matrix_ = cmn.add_counters_to_axis(
            X=TFIDF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 5.-- Correspondence Analysis
        #
        ca = CA()

        ca.fit(TFIDF_matrix_)

        self.eigenvalues_ = ca.eigenvalues_
        self.explained_variance_ = ca.explained_variance_
        self.principal_coordinates_rows_ = ca.principal_coordinates_rows_
        self.principal_coordinates_cols_ = ca.principal_coordinates_cols_

        #
        # 6.-- Selects the first n_factors to cluster
        #
        X = self.principal_coordinates_cols_[
            self.principal_coordinates_cols_.columns[0 : self.n_factors]
        ]
        X = pd.DataFrame(
            X,
            columns=["dim-{}".format(i) for i in range(self.n_factors)],
            index=TFIDF_matrix_.columns,
        )

        #
        # 7.-- Cluster analysis of first n_factors of CA matrix
        #
        if self.clustering_method == "Affinity Propagation":
            labels = AffinityPropagation(
                random_state=int(self.random_state)
            ).fit_predict(X)
            self.n_clusters = len(set(labels))

        if self.clustering_method == "Agglomerative Clustering":
            labels = AgglomerativeClustering(
                n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage
            ).fit_predict(X)

        if self.clustering_method == "Birch":
            labels = Birch(n_clusters=self.n_clusters).fit_predict(X)

        if self.clustering_method == "DBSCAN":
            labels = DBSCAN().fit_predict(X)
            self.n_clusters = len(set(labels))

        #  if self.clustering_method == "Feature Agglomeration":
        #      m = FeatureAgglomeration(
        #          n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage
        #      ).fit(X)
        #      labels = ???

        if self.clustering_method == "KMeans":
            labels = KMeans(
                n_clusters=self.n_clusters, random_state=int(self.random_state)
            ).fit_predict(X)

        if self.clustering_method == "Mean Shift":
            labels = MeanShift().fit_predct(X)
            self.n_clusters = len(set(labels))

        #
        # 8.-- Cluster centers
        #
        X["CLUSTER"] = labels
        self.cluster_centers_ = X.groupby("CLUSTER").mean()

        #
        # 9.-- Memberships
        #
        communities = pd.DataFrame(
            pd.NA, columns=range(self.n_clusters), index=range(self.top_n)
        )
        for i_cluster in range(self.n_clusters):
            R = X[X.CLUSTER == i_cluster]
            R = cmn.sort_axis(data=R, num_documents=True, axis=0, ascending=False,)
            community = R.index
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
        self.memberships_ = communities

        #
        # 10.-- Cluster names (most frequent term in cluster)
        #
        names = communities.loc[0, :].tolist()
        self.cluster_names_ = names

    def cluster_names(self):
        self.apply()
        return self.cluster_names_

    def cluster_centers(self):
        self.apply()
        return self.cluster_centers_

    def memberships(self):
        self.apply()
        return self.memberships_

    def plot_singular_values(self):
        self.apply()
        return plt.barh(width=self.eigenvalues_[:20])

    def plot_clusters(self):

        self.apply()

        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()

        x = self.cluster_centers_["dim-{}".format(self.x_axis)]
        y = self.cluster_centers_["dim-{}".format(self.y_axis)]
        names = self.cluster_names_
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

        pos = {term: (x[idx], y[idx]) for idx, term in enumerate(self.cluster_names_)}
        cmn.ax_text_node_labels(
            ax=ax, labels=self.cluster_names_, dict_pos=pos, node_sizes=node_sizes
        )

        cmn.ax_expand_limits(ax)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(
            x=xlim[1],
            y=0.01 * (ylim[1] - ylim[0]),
            s="Dim-{}".format(self.x_axis),
            fontsize=9,
            color="dimgray",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.text(
            x=0.01 * (xlim[1] - xlim[0]),
            y=ylim[1],
            s="Dim-{}".format(self.y_axis),
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


COLUMNS = [
    "Author_Keywords",
    "Index_Keywords",
    "Abstract_words_CL",
    "Abstract_words",
    "Title_words_CL",
    "Title_words",
    "Author_Keywords_CL",
    "Index_Keywords_CL",
]


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.app_title = "Comparative analysis"
        self.menu_options = [
            "Cluster names",
            "Cluster centers",
            "Memberships",
            "Plot singular values",
            "Plot clusters",
        ]

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_widgets = [
            dash.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Clustering"),
            dash.dropdown(desc="N Factors:", options=list(range(2, 11)),),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.random_state(),
            dash.separator(text="Visualization"),
            dash.top_n(m=10, n=51, i=5),
            dash.cmap(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        self.panel_widgets[-3]["widget"].options = [i for i in range(self.n_factors)]
        self.panel_widgets[-4]["widget"].options = [i for i in range(self.n_factors)]

        #

        for i in [-1, -2, -3, -4, -5]:
            self.panel_widgets[i]["widget"].disabled = (
                True
                if self.menu in ["Cluster names", "Cluster centers", "Memberships"]
                else False
            )

        for i in [-6]:
            self.panel_widgets[i]["widget"].disabled = (
                False if self.menu in ["Memberships"] else True
            )

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


def app(
    data, limit_to=None, exclude=None, years_range=None,
):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

