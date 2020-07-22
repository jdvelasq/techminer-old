import json

import pandas as pd
from sklearn.cluster import KMeans

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
    def __init__(self, data, limit_to, exclude):
        #
        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude
        #
        self.analysis_type = None
        self.ascending = None
        self.cmap = None
        self.column = None
        self.height = None
        self.max_iter = None
        self.max_iters = None
        self.max_terms = None
        self.min_occurrence = None
        self.n_clusters = None
        self.n_components = None
        self.n_factors = None
        self.n_iter = None
        self.norm = None
        self.random_state = None
        self.smooth_idf = None
        self.sort_by = None
        self.sublinear_tf = None
        self.top_by = None
        self.top_n = None
        self.top_n = None
        self.use_idf = None
        self.width = None
        self.x_axis = None
        self.y_axis = None

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
        # 2.-- Computtes TFIDF matrix and select max_term frequent terms
        #
        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            max_terms=self.max_terms,
        )

        TFIDF_matrix_ = cmn.add_counters_to_axis(
            X=TFIDF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 3.-- Correspondence Analysis
        #
        ca = CA()

        ca.fit(TFIDF_matrix_)

        self.eigenvalues_ = ca.eigenvalues_
        self.explained_variance_ = ca.explained_variance_
        self.principal_coordinates_rows_ = ca.principal_coordinates_rows_
        self.principal_coordinates_cols_ = ca.principal_coordinates_cols_

        #
        # 4.-- Cluster analysis of first n_factors of CA matrix
        #
        X = self.principal_coordinates_cols_[
            self.principal_coordinates_cols_.columns[0 : self.n_factors]
        ]
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=int(self.random_state),
        )
        kmeans.fit(X)
        self.cluster_centers_ = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=self.principal_coordinates_cols_.columns[0 : self.n_factors],
        )

        #
        # 5.-- Memberships
        #
        self.memberships_ = pd.DataFrame(
            kmeans.labels_,
            columns=["Cluster"],
            index=self.principal_coordinates_cols_.index,
        )
        self.memberships_ = self.memberships_.sort_values("Cluster", ascending=True)

        #
        # 6.-- Cluster names (most frequent term in cluster)
        #
        cluster_names = []
        for i_cluster in range(self.n_clusters):
            cluster_members = self.memberships_[self.memberships_.Cluster == i_cluster]
            cluster_members = cmn.sort_axis(
                data=cluster_members, num_documents=True, axis=0, ascending=False
            )
            cluster_names.append(cluster_members.index[0])
        self.cluster_names_ = pd.DataFrame(cluster_names, columns=["Name"])

    def cluster_names(self):
        self.apply()
        return self.cluster_names_

    def cluster_centers(self):
        self.apply()
        return self.cluster_centers_

    def memberships(self):
        self.apply()
        r = {}
        for i_cluster in range(self.n_clusters):
            cluster_members = self.memberships_[self.memberships_.Cluster == i_cluster]
            cluster_members = cmn.sort_axis(
                data=cluster_members, num_documents=True, axis=0, ascending=False
            )
            m = cluster_members.head(self.top_n)
            m = m.index.tolist()
            r[i_cluster] = m
        r = json.dumps(r, indent=4, sort_keys=True)
        print(r)
        return ""

    def plot_singular_values(self):
        self.apply()
        return plt.barh(width=self.eigenvalues_)

    def plot_clusters(self):

        self.apply()

        return diagram_plot(
            x=self.cluster_centers_[self.cluster_centers_.columns[self.x_axis]],
            y=self.cluster_centers_[self.cluster_centers_.columns[self.y_axis]],
            labels=self.cluster_names_["Name"].tolist(),
            x_axis_at=0,
            y_axis_at=0,
            cmap=self.cmap,
            width=self.width,
            height=self.height,
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
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

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
            dash.max_terms(),
            dash.n_clusters(),
            dash.max_iter(),
            dash.random_state(),
            dash.dropdown(desc="N Factors:", options=list(range(2, 20)),),
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


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(
    data,
    limit_to=None,
    exclude=None,
    norm=None,
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False,
):
    return DASHapp(
        data=data,
        limit_to=limit_to,
        exclude=exclude,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    ).run()
