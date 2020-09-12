import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.manifold import MDS
import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    limit_to_exclude,
    normalize_network,
    sort_axis,
    cluster_table_to_list,
    cluster_table_to_python_code,
    keywords_coverage,
)
from techminer.plots import (
    conceptual_structure_map,
    counters_to_node_sizes,
    xy_clusters_plot,
)

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

    def apply_mds(self):

        ##
        ## Conceptual Structure Map using Multidimensinal Scaling
        ##

        X = self.data.copy()

        ##
        ## Compute TF matrix
        ##
        TF_matrix_ = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occurrence,
        )

        ##
        ## Limit to / Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## Add counters to axes
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ## Select top terms
        ##
        TF_matrix_ = sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        ##
        ## Co-occurrence matrix and normalization using Association Index
        ##
        M = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        M = pd.DataFrame(M, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        M = normalize_network(M, normalization="Association")

        ##
        ## Dissimilarity matrix
        ##
        M = 1 - M
        for i in M.columns.tolist():
            M.at[i, i] = 0.0

        ##
        ## Decomposition
        ##
        model = MDS(
            n_components=2,
            random_state=int(self.random_state),
            dissimilarity="precomputed",
        )

        R = model.fit_transform(X=M.values)
        R = pd.DataFrame(
            R,
            columns=["Dim-{}".format(i) for i in range(2)],
            index=M.columns,
        )

        ##
        ## Clustering
        ##
        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=R,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

        R["Cluster"] = self.labels_

        self.coordinates_ = R

        ##
        ## Results
        ##
        self.X_ = R

    def keywords_map(self):
        self.apply()
        X = self.X_
        cluster_labels = X.Cluster
        X.pop("Cluster")
        return conceptual_structure_map(
            coordinates=X,
            cluster_labels=cluster_labels,
            top_n=self.top_n,
            figsize=(self.width, self.height),
        )

    def keywords_by_cluster_table(self):
        self.apply()
        return self.cluster_members_

    def keywords_by_cluster_list(self):
        self.apply()
        return cluster_table_to_list(self.cluster_members_)

    def keywords_by_cluster_python_code(self):
        self.apply()
        return cluster_table_to_python_code(self.column, self.cluster_members_)

    def keywords_coverage(self):

        X = self.data.copy()

        ##
        ## Compute TF matrix
        ##
        TF_matrix_ = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occurrence,
        )

        ##
        ## Limit to / Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## Add counters to axes
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ## Select top terms
        ##
        TF_matrix_ = sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]

        ##
        ## Keywords list
        ##
        keywords_list = TF_matrix_.columns

        return keywords_coverage(
            data=self.data, column=self.column, keywords_list=keywords_list
        )

    def apply(self):

        if self.method == "Multidimensional Scaling":
            self.apply_mds()

    def XXX_apply(self):
        #
        X = self.data.copy()

        #
        # 1.-- TF matrix
        #
        TF_matrix_ = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
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
        # 3.-- Add counters to axes
        #
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 4.-- Select top terms
        #
        TF_matrix_ = sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        #
        # 5.-- Co-occurrence matrix and normalization
        #
        M = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        M = pd.DataFrame(M, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        M = normalize_network(M, normalization=self.normalization)

        #
        # 6.-- Dissimilarity matrix
        #
        M = 1 - M
        for i in M.columns.tolist():
            M.at[i, i] = 0.0

        #
        # 5.-- Number of factors
        #
        # self.n_components = 2 if self.decomposition_method == "MDS" else 10

        #
        # 6.-- Factor decomposition
        #
        model = {
            "Factor Analysis": FactorAnalysis,
            "PCA": PCA,
            "Fast ICA": FastICA,
            "SVD": TruncatedSVD,
            "MDS": MDS,
        }[self.decomposition_method]

        model = (
            model(
                n_components=self.n_components,
                random_state=int(self.random_state),
                dissimilarity="precomputed",
            )
            if self.decomposition_method == "MDS"
            else model(
                n_components=self.n_components, random_state=int(self.random_state)
            )
        )

        R = model.fit_transform(X=M.values)
        R = pd.DataFrame(
            R,
            columns=["Dim-{}".format(i) for i in range(self.n_components)],
            index=M.columns,
        )

        #
        # 7.-- Clustering
        #
        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=R,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

        R["Cluster"] = self.labels_

        self.coordinates_ = R

        #
        # 8.-- Results
        #
        self.X_ = R

    def XXX_cluster_members(self):
        self.apply()
        return self.cluster_members_

    def XXX_conceptual_structure_map(self):
        self.apply()
        X = self.X_
        cluster_labels = X.Cluster
        X.pop("Cluster")
        return conceptual_structure_map(
            coordinates=X,
            cluster_labels=cluster_labels,
            top_n=self.top_n,
            figsize=(self.width, self.height),
        )

    def XXX_conceptual_structure_members(self):
        self.apply()
        return self.cluster_members_

    def XXX_cluster_plot(self):
        self.apply()
        return xy_clusters_plot(
            x=self.cluster_centers_["Dim-{}".format(self.x_axis)],
            y=self.cluster_centers_["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=self.cluster_names_,
            node_sizes=counters_to_node_sizes(self.cluster_names_),
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

COLUMNS = sorted(
    [
        "Abstract_phrase_words",
        "Abstract_words_CL",
        "Abstract_words",
        "Author_Keywords_CL",
        "Author_Keywords",
        "Index_Keywords_CL",
        "Index_Keywords",
        "Keywords_CL",
        "Title_words_CL",
        "Title_words",
    ]
)


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

        self.app_title = "Conceptual Structure"
        self.menu_options = [
            "Keywords Map",
            "Keywords by Cluster (table)",
            "Keywords by Cluster (list)",
            "Keywords by Cluster (Python code)",
            "Keywords coverage",
            "Most Cited Papers by Cluster",
        ]
        #
        self.panel_widgets = [
            dash.dropdown(
                desc="Method:",
                options=["Multidimensional Scaling"],
            ),
            dash.dropdown(
                desc="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.random_state(),
            dash.separator(text="Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=2, n=11, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.separator(text="Visualization"),
            dash.top_n(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Cluster members":
            #
            self.n_components = 10
            #
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")

        if self.menu == "Cluster plot":
            #
            self.n_components = 10
            #
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu in ["Conceptual Structure Map", "Conceptual Structure Members"]:
            #
            self.n_components = 2
            #
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        self.enable_disable_clustering_options()


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def conceptual_structure(
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
