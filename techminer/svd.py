import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import techminer.plots as plt
import techminer.common as cmn
from techminer.dashboard import DASH
from techminer.document_term import TF_matrix, TFIDF_matrix
from techminer.params import EXCLUDE_COLS


from techminer.diagram_plot import diagram_plot
import techminer.dashboard as dash

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
        # 3.-- Data to analyze
        #
        X = None
        if self.analysis_type == "Co-occurrence":
            X = np.matmul(TFIDF_matrix_.transpose().values, TFIDF_matrix_.values)
            X = pd.DataFrame(
                X, columns=TFIDF_matrix_.columns, index=TFIDF_matrix_.columns
            )
        if self.analysis_type == "TF*IDF":
            X = TFIDF_matrix_

        #
        # 4.-- SVD for a maximum of 20 dimensions
        #
        TruncatedSVD_ = TruncatedSVD(
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=int(self.random_state),
        ).fit(X)

        #
        # 5.-- Results
        #
        axis_names = ["dim-{:>02d}".format(i) for i in range(self.n_components)]
        self.components_ = pd.DataFrame(
            np.transpose(TruncatedSVD_.components_),
            columns=axis_names,
            index=X.columns,
        )
        self.statistics_ = pd.DataFrame(
            TruncatedSVD_.explained_variance_,
            columns=["Explained Variance"],
            index=axis_names,
        )
        self.statistics_["Explained Variance"] = TruncatedSVD_.explained_variance_
        self.statistics_[
            "Explained Variance Ratio"
        ] = TruncatedSVD_.explained_variance_ratio_
        self.statistics_["Singular Values"] = TruncatedSVD_.singular_values_

    def table(self):
        self.apply()
        X = self.components_
        X = cmn.limit_to_exclude(
            data=X,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        X = cmn.sort_by_axis(data=X, sort_by=self.top_by, ascending=False, axis=0)
        X = X.head(self.top_n)
        X = cmn.sort_by_axis(
            data=X, sort_by=self.sort_by, ascending=self.ascending, axis=0
        )
        return X

    def statistics(self):
        self.apply()
        return self.statistics_

    def plot_singular_values(self):

        self.apply()

        return plt.barh(width=self.statistics_["Singular Values"])

    def plot_relationships(self):

        self.apply()
        X = self.table()

        return diagram_plot(
            x=X[X.columns[self.x_axis]],
            y=X[X.columns[self.y_axis]],
            labels=X.index,
            x_axis_at=0,
            y_axis_at=0,
            cmap=self.cmap,
            width=self.width,
            height=self.height,
        )

        # matplotlib.rc("font", size=11)
        # fig = pyplot.Figure(figsize=(self.width, self.height))
        # ax = fig.subplots()
        # cmap = pyplot.cm.get_cmap(self.cmap)

        # X = self.table()
        # X.columns = X.columns.tolist()

        # node_sizes = cmn.counters_to_node_sizes(x=X.index)
        # node_colors = cmn.counters_to_node_colors(x=X.index, cmap=cmap)

        # ax.scatter(
        #     X[X.columns[self.x_axis]],
        #     X[X.columns[self.y_axis]],
        #     s=node_sizes,
        #     linewidths=1,
        #     edgecolors="k",
        #     c=node_colors,
        # )

        # cmn.ax_expand_limits(ax)
        # cmn.ax_text_node_labels(
        #     ax,
        #     labels=X.index,
        #     dict_pos={
        #         key: (c, d)
        #         for key, c, d in zip(
        #             X.index, X[X.columns[self.x_axis]], X[X.columns[self.y_axis]],
        #         )
        #     },
        #     node_sizes=node_sizes,
        # )

        # ax.axhline(
        #     y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
        # )
        # ax.axvline(
        #     x=0, color="gray", linestyle="--", linewidth=1, zorder=-1,
        # )

        # ax.axis("off")

        # cmn.set_ax_splines_invisible(ax)

        # fig.set_tight_layout(True)

        # return fig


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
            dash.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            dash.dropdown(desc="Analysis type:", options=["Co-occurrence", "TF*IDF",],),
            dash.n_components(),
            dash.random_state(),
            dash.n_iter(),
            dash.min_occurrence(),
            dash.max_terms(),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            dash.top_n(),
            dash.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Num Documents", "Times Cited",],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
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
