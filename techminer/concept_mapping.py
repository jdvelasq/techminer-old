import ipywidgets as widgets

import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

import techminer.common as cmn
import techminer.dashboard as dash
from techminer.correspondence import CA
from techminer.dashboard import DASH
from techminer.document_term import TF_matrix, TFIDF_matrix
from techminer.graph import network_normalization


def network_clustering(X, n_clusters, affinity, linkage):

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, affinity=affinity, linkage=linkage
    )
    clustering.fit(1 - X)
    return {key: value for key, value in zip(X.columns, clustering.labels_)}


def cluster_plot(X, method, n_components, x_axis, y_axis, figsize):

    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    n_clusters = len(X.columns)
    #
    # Both methods use the dissimilitud matrix
    #
    if method == "MDS":
        embedding = MDS(n_components=n_components)
        X_transformed = embedding.fit_transform(1 - X,)
        x_axis = X_transformed[:, x_axis]
        y_axis = X_transformed[:, y_axis]
    if method == "CA":
        ca = CA()
        ca.fit(1 - X)
        X_transformed = ca.principal_coordinates_cols_
        x_axis = X_transformed.loc[:, X_transformed.columns[x_axis]]
        y_axis = X_transformed.loc[:, X_transformed.columns[y_axis]]

    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = pyplot.cm.get_cmap(cmap_name)
        colors += [cmap(0.025 + 0.05 * i) for i in range(20)]

    node_sizes = cmn.counters_to_node_sizes(X.columns)

    node_colors = [cmap(0.2 + 0.80 * t / (n_clusters - 1)) for t in range(n_clusters)]

    ax.scatter(
        x_axis, y_axis, s=node_sizes, linewidths=1, edgecolors="k", c=node_colors
    )

    cmn.ax_expand_limits(ax)

    pos = {term: (x_axis[idx], y_axis[idx]) for idx, term in enumerate(X.columns)}
    cmn.ax_text_node_labels(
        ax=ax, labels=X.columns, dict_pos=pos, node_sizes=node_sizes
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)

    ax.set_aspect("equal")
    ax.axis("off")
    cmn.set_ax_splines_invisible(ax)

    fig.set_tight_layout(True)

    return fig


def strategic_map(centrality, density, cluster_names, cluster_co_occurrence, figsize):

    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    min_co_occurrence = min(cluster_co_occurrence)
    max_co_occurrence = max(cluster_co_occurrence)

    node_sizes = 500 + 2500 * (cluster_co_occurrence - min_co_occurrence) / (
        max_co_occurrence - min_co_occurrence
    )
    node_sizes = node_sizes.tolist()

    cmap = pyplot.cm.get_cmap("tab20")

    median_centrality = centrality.median()
    median_density = density.median()

    node_colors = []
    for i_cluster, _ in enumerate(centrality):
        if centrality[i_cluster] > median_centrality:
            if density[i_cluster] >= median_density:
                node_colors.append(cmap(0.0))
            else:
                node_colors.append(cmap(0.25))
        else:
            if density[i_cluster] >= median_density:
                node_colors.append(cmap(0.50))
            else:
                node_colors.append(cmap(0.75))

    ax.scatter(
        centrality, density, s=node_sizes, linewidths=1, edgecolors="k", c=node_colors,
    )

    cmn.ax_expand_limits(ax)
    cmn.ax_text_node_labels(
        ax,
        labels=cluster_names,
        dict_pos={key: (c, d) for key, c, d in zip(cluster_names, centrality, density)},
        node_sizes=node_sizes,
    )

    ax.axhline(
        y=median_density, color="gray", linestyle="--", linewidth=1, zorder=-1,
    )
    ax.axvline(
        x=median_centrality, color="gray", linestyle="--", linewidth=1, zorder=-1,
    )

    #  ax.set_aspect("equal")
    ax.axis("off")

    cmn.set_ax_splines_invisible(ax)

    fig.set_tight_layout(True)

    return fig


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
        ## Concept mapping
        ## https://tlab.it/en/allegati/help_en_online/mmappe2.htm
        ##

        #
        # 1.-- Co-occurrence matrix
        #
        #
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occurrence,
        )

        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=None,
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
            max_items=self.max_items,
        )

        TFIDF_matrix_ = cmn.add_counters_to_axis(
            X=TFIDF_matrix_, axis=1, data=self.data, column=self.column
        )

        X = np.matmul(TFIDF_matrix_.transpose().values, TFIDF_matrix_.values)
        X = pd.DataFrame(X, columns=TFIDF_matrix_.columns, index=TFIDF_matrix_.columns)

        #
        # 2.-- Association indexes
        #
        X = network_normalization(X=X, normalization=self.normalization)

        #
        # 3.-- Hierarchical clustering of the dissimilarity matrix
        #
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage
        )
        clustering.fit(1 - X)
        clusters_dict = {
            key: value for key, value in zip(X.columns, clustering.labels_)
        }

        #
        # 4.-- Cluster membership
        #
        memberships = {key: [] for key in range(self.n_clusters)}
        for t in X.columns:
            cluster = clusters_dict[t]
            memberships[cluster] += [t]
        for cluster in range(self.n_clusters):
            memberships[cluster] = sorted(
                memberships[cluster], key=lambda w: w.split(" ")[-1], reverse=True
            )

        self.memberships_ = memberships

        #
        # 5.-- Cluster co-occurrence
        #
        M = X.copy()
        M["CLUSTER"] = M.index
        M["CLUSTER"] = M["CLUSTER"].map(lambda t: clusters_dict[t])
        M = M.groupby("CLUSTER").sum()
        M = M.transpose()

        M["CLUSTER"] = M.index
        M["CLUSTER"] = M["CLUSTER"].map(lambda t: clusters_dict[t])
        M = M.groupby("CLUSTER").sum()
        M.columns = ["Cluster {}".format(i) for i in range(self.n_clusters)]
        M.index = M.columns
        self.cluster_co_occurrence_ = M

        #
        # 6.-- Strategic Map
        #
        cluster_names = [memberships[cluster][0] for cluster in range(self.n_clusters)]
        str_map = pd.DataFrame(cluster_names, columns=["Cluster name"], index=M.columns)

        str_map["Density"] = 0.0
        str_map["Centrality"] = 0.0

        ## Density -- internal conections
        for cluster in M.columns:
            str_map.at[cluster, "Density"] = M[cluster][cluster]

        ## Centrality -- outside conections
        S = M.sum()
        str_map["Centrality"] = S
        str_map["Centrality"] = str_map["Centrality"] - str_map["Density"]

        self.centrality_density_ = str_map

    def cluster_memberships(self):
        self.apply()
        text = []
        for key in self.memberships_:
            text += ["=" * 60]
            text += [str(key)]
            for t in self.memberships_[key]:
                text += ["      {:>50}".format(t)]
        return widgets.HTML("\n".join(text))

    def cluster_co_occurrence_matrix(self):
        self.apply()
        return self.cluster_co_occurrence_

    def mds_cluster_map(self):
        self.apply()
        return cluster_plot(
            X=self.cluster_co_occurrence_,
            method="MDS",
            n_components=self.n_components,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            figsize=((self.width, self.height)),
        )

    def ca_cluster_map(self):
        self.apply()
        return cluster_plot(
            X=self.cluster_co_occurrence_,
            method="CA",
            n_components=self.n_components,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            figsize=(self.width, self.height),
        )

    def centratlity_density_table(self):
        self.apply()
        return self.centrality_density_

    def strategic_map(self):
        self.apply()
        cluster_co_occurrence = [
            self.cluster_co_occurrence_[i][i]
            for i in self.cluster_co_occurrence_.columns
        ]
        return strategic_map(
            centrality=self.centrality_density_.Centrality,
            density=self.centrality_density_.Density,
            cluster_names=self.centrality_density_["Cluster name"],
            cluster_co_occurrence=cluster_co_occurrence,
            figsize=(self.width, self.height),
        )


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
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.app_title = "Concept Mapping"
        self.menu_options = [
            "Cluster memberships",
            "Cluster co-occurrence matrix",
            "Centratlity-Density table",
            "MDS cluster map",
            "CA cluster map",
            "Strategic map",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.normalization(),
            dash.separator(text="Aglomerative Clustering"),
            dash.n_clusters(),
            dash.affinity(),
            dash.linkage(),
            dash.separator(text="MDS/CA diagram"),
            dash.n_components(),
            dash.x_axis(),
            dash.y_axis(),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            dash.top_n(),
            dash.fig_width(),
            dash.fig_height(),
        ]

        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        with self.output:

            if self.menu in [
                "Cluster memberships",
                "Cluster co-occurrence matrix",
                "Centratlity-Density table",
            ]:
                self.set_disabled("Width:")
                self.set_disabled("Height:")
            else:
                self.set_enabled("Width:")
                self.set_enabled("Height:")

            if self.menu in [
                "MDS cluster map",
                "CA cluster map",
            ]:
                self.set_enabled("X-axis:")
                self.set_enabled("Y-axis:")
            else:
                self.set_disabled("X-axis:")
                self.set_disabled("Y-axis:")

            if self.menu == "MDS cluster map":

                self.panel_widgets[-5]["widget"].disabled = False
                self.panel_widgets[8]["widget"].options = list(range(self.n_components))
                self.panel_widgets[9]["widget"].options = list(range(self.n_components))
            else:
                self.panel_widgets[-5]["widget"].disabled = True

            if self.menu == "CA cluster map":
                self.panel_widgets[8]["widget"].options = list(range(self.n_clusters))
                self.panel_widgets[9]["widget"].options = list(range(self.n_clusters))


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

