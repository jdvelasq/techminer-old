from collections import Counter
from techminer.core import CA
from techminer.core import add_counters_to_axis
from techminer.core import clustering
from techminer.core import DASH
from techminer.core import limit_to_exclude
from techminer.core import TF_matrix, TFIDF_matrix
from techminer.core.params import EXCLUDE_COLS
from techminer.plots import counters_to_node_sizes
from techminer.plots import xy_clusters_plot
import pandas as pd
import techminer.core.dashboard as dash
from techminer.core import corpus_filter

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
        limit_to,
        exclude,
        years_range,
        clusters=None,
        cluster=None,
    ):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        #
        # Filter for cluster members
        #
        if clusters is not None and cluster is not None:
            data = corpus_filter(data=data, clusters=clusters, cluster=cluster)

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def correspondence_analysis(self):

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
            scheme="binary",
            min_occurrence=self.min_occurrence,
        )

        #
        # 2.-- Limit to / Exclude
        #
        TF_matrix_ = limit_to_exclude(
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
        TFIDF_matrix_ = add_counters_to_axis(
            X=TFIDF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 5.-- Correspondence Analysis
        #      10 first factors for ploting
        #
        ca = CA()

        ca.fit(TFIDF_matrix_)

        self.eigenvalues_ = ca.eigenvalues_[0:10]
        self.explained_variance_ = ca.explained_variance_[0:10]

        z = ca.principal_coordinates_rows_
        z = z[z.columns[:10]]
        self.principal_coordinates_rows_ = z

        z = ca.principal_coordinates_cols_
        z = z[z.columns[:10]]
        self.principal_coordinates_cols_ = z

        #
        # 6.-- Correspondence analysis plot
        #
        self.correspondence_analysis_plot_coordinates_ = (
            self.principal_coordinates_cols_
        )
        self.correspondence_analysis_plot_labels_ = TFIDF_matrix_.columns

        self.TFIDF_matrix_ = TFIDF_matrix_

    def make_clustering(self):

        #
        # 1.-- Selects the first n_factors to cluster
        #
        X = self.principal_coordinates_cols_[
            self.principal_coordinates_cols_.columns[0 : self.n_factors]
        ]
        X = pd.DataFrame(
            X,
            columns=["Dim-{}".format(i) for i in range(self.n_factors)],
            index=self.TFIDF_matrix_.columns,
        )

        #
        # 2.-- Cluster analysis of first n_factors of CA matrix
        #
        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=X,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

        X["CLUSTER"] = self.labels_

    def correspondence_analysis_plot(self):
        #
        self.correspondence_analysis()
        #
        coordinates = self.correspondence_analysis_plot_coordinates_.head(self.top_n)

        return xy_clusters_plot(
            x=coordinates["Dim-{}".format(self.x_axis)],
            y=coordinates["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=self.correspondence_analysis_plot_labels_[: self.top_n],
            node_sizes=counters_to_node_sizes(coordinates.index),
            color_scheme=self.color_scheme,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )

    def cluster_members(self):
        #
        self.correspondence_analysis()
        self.make_clustering()
        #
        return self.cluster_members_

    def cluster_plot(self):
        #
        self.correspondence_analysis()
        self.make_clustering()
        #
        labels = self.cluster_members_.loc[0, :].tolist()
        labels = [
            "CLUST_{} {}".format(index, label) for index, label in enumerate(labels)
        ]
        #
        node_sizes = Counter(self.labels_)
        node_sizes = [node_sizes[i] for i in range(len(node_sizes.keys()))]
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        node_sizes = [
            500 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]
        #
        return xy_clusters_plot(
            x=self.cluster_centers_["Dim-{}".format(self.x_axis)],
            y=self.cluster_centers_["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=labels,
            node_sizes=node_sizes,
            color_scheme=self.color_scheme,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )


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
    def __init__(
        self,
        data,
        limit_to=None,
        exclude=None,
        years_range=None,
        clusters=None,
        cluster=None,
    ):
        """Dashboard app"""

        Model.__init__(
            self,
            data=data,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
            clusters=clusters,
            cluster=cluster,
        )
        DASH.__init__(self)

        self.app_title = "Comparative analysis"
        self.menu_options = [
            "Correspondence analysis plot",
            "Cluster members",
            "Cluster plot",
        ]

        COLUMNS = sorted(
            [column for column in sorted(data.columns) if column not in EXCLUDE_COLS]
        )

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:",
                options=[t for t in sorted(data.columns) if t in COLUMNS],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Clustering"),
            dash.dropdown(
                desc="N Factors:",
                options=list(range(2, 11)),
            ),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.random_state(),
            dash.separator(text="Visualization"),
            dash.top_n(m=10, n=51, i=5),
            dash.color_scheme(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        def visualization_disabled():

            self.set_disabled("Color Scheme:")
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        def visualization_enabled():

            self.set_enabled("Color Scheme:")
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        def clustering_disabled():

            self.set_disabled("N Factors:")
            self.set_disabled("Clustering Method:")
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_disabled("Random State:")

        def clustering_enabled():

            self.set_enabled("N Factors:")
            self.set_enabled("Clustering Method:")
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            self.set_enabled("Random State:")

            self.enable_disable_clustering_options(include_random_state=True)

        if self.menu == "Correspondence analysis plot":

            clustering_disabled()
            visualization_enabled()

        if self.menu == "Cluster members":

            clustering_enabled()
            visualization_disabled()

        if self.menu == "Cluster plot":

            clustering_enabled()
            visualization_enabled()


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def comparative_analysis(
    input_file="techminer.csv",
    limit_to=None,
    exclude=None,
    years_range=None,
    clusters=None,
    cluster=None,
):
    return DASHapp(
        data=pd.read_csv(input_file),
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
        clusters=clusters,
        cluster=cluster,
    ).run()
