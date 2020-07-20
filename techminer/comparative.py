import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.pyplot as pyplot

import techminer.plots as plt
import techminer.common as cmn
from techminer.dashboard import DASH
from techminer.document_term import TF_matrix, TFIDF_matrix
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS

import techminer.gui as gui

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
        self.max_terms = None
        self.min_occurrence = None
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
        self.use_idf = None
        self.width = None
        self.x_axis = None
        self.y_axis = None

    def apply(self):

        ##
        ## SVD for documents x terms matrix & co-occurrence matrix
        ##   from https://tlab.it/en/allegati/help_en_online/msvd.htm
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
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        kmeans.fit(X)
        self.cluster_centers_ = pd.DataFrame(
            kmeans.cluster_centers_, columns=self.principal_coordinates_cols_.columns
        )

        #
        # 5.-- Memberships
        #
        self.memberships_ = pd.DataFrame(
            kmeans.labels_,
            columns=["Cluster"],
            index=self.self.principal_coordinates_cols_.index,
        )

        #
        # 6.-- Cluster names (most frequent term in cluster)
        #
        cluster_names = []
        for i_cluster in range(self.n_clusters):
            cluster_members = self.memberships_[self.memberships_ == i_cluster]
            cluster_members = cmn.sort_axis(
                data=cluster_members, num_documents=True, axis=0, ascending=False
            )
            cluster_names.append(cluster_members.index[0])
        self.cluster_names_ = pd.DataFrame(cluster_names, columns=["Name"])

    def cluster_names(self):
        return self.cluster_names_

    def cluster_centers(self):
        return self.cluster_centers_

    def plot_singular_values(self):
        self.apply()
        return plt.barh(width=self.statistics_["Singular Values"])

    def plot_terms(self):

        self.apply()

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.cmap)

        X = self.principal_coordinates_cols_()
        X.columns = X.columns.tolist()

        node_sizes = cmn.counters_to_node_sizes(x=X.index)
        node_colors = cmn.counters_to_node_colors(x=X.index, cmap=cmap)

        ax.scatter(
            X[X.columns[self.x_axis]],
            X[X.columns[self.y_axis]],
            s=node_sizes,
            linewidths=1,
            edgecolors="k",
            c=node_colors,
        )

        cmn.ax_expand_limits(ax)
        cmn.ax_text_node_labels(
            ax,
            labels=X.index,
            dict_pos={
                key: (c, d)
                for key, c, d in zip(
                    X.index, X[X.columns[self.x_axis]], X[X.columns[self.y_axis]],
                )
            },
            node_sizes=node_sizes,
        )

        ax.axhline(
            y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
        )
        ax.axvline(
            x=0, color="gray", linestyle="--", linewidth=1, zorder=-1,
        )

        ax.axis("off")

        cmn.set_ax_splines_invisible(ax)

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

        self.app_title = "SVD"
        self.menu_options = [
            "Table",
            "Statistics",
            "Plot singular values",
            "Plot relationships",
        ]

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_widgets = [
            gui.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            gui.dropdown(desc="Analysis type:", options=["Co-occurrence", "TF*IDF",],),
            gui.n_components(),
            gui.random_state(),
            gui.n_iter(),
            gui.min_occurrence(),
            gui.max_terms(),
            gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            gui.top_n(),
            gui.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            gui.ascending(),
            gui.cmap(),
            gui.x_axis(),
            gui.y_axis(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        self.panel_widgets[-3]["widget"].options = [
            i for i in range(self.panel_widgets[2]["widget"].value)
        ]
        self.panel_widgets[-4]["widget"].options = [
            i for i in range(self.panel_widgets[2]["widget"].value)
        ]

        #

        for i in [-1, -2, -3, -4, -5]:
            self.panel_widgets[i]["widget"].disabled = (
                True if self.menu in ["Table", "Statistics"] else False
            )

        for i in [-6, -7, -8, -9]:
            self.panel_widgets[i]["widget"].disabled = (
                False if self.menu in ["Table", "Statistics"] else True
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
