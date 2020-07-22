import matplotlib
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.cluster import KMeans

import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.correspondence import CA
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
        ##
        self.ascending = None
        self.cmap = None
        self.column = None
        self.height = None
        self.max_iter = None
        self.n_clusters = None
        self.norm = None
        self.smooth_idf = None
        self.sort_by = None
        self.sublinear_tf = None
        self.top_by = None
        self.top_n = None
        self.use_idf = None
        self.width = None
        self.x_axis = None
        self.y_axis = None
        ##

    def fit(self):

        ##
        ## Fuente:
        ##   https://tlab.it/en/allegati/help_en_online/mrepert.htm
        ##
        #
        # 1.-- Construye TF_matrix binaria
        #
        TF_matrix_ = TF_matrix(self.data, self.column, scheme="binary")
        TF_matrix_ = cmn.limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        TF_matrix_ = cmn.add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )
        #  TF_matrix_ = cmn.sort_by_axis(
        #      data=TF_matrix_, sort_by=self.top_by, ascending=False, axis=1
        #  )
        #  TF_matrix_ = TF_matrix_[TF_matrix_.columns[: self.top_n]]

        #
        # 2.-- Construye TF-IDF y escala filas to longitud unitaria (norma euclidiana).
        #      En sklearn: norm='l2'
        #
        #      En TLAB se usa tf * log( N / df).
        #      Para sklearn:
        #        * use_idf = False
        #        * sublinear_tf = False
        #
        TF_IDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )

        #
        # 3.-- Clustering de las filas de TF_IDF_matrix_.
        #      En TLAB se usa bisecting k-means.
        #      Se implementa sklearn.cluster.KMeans
        #
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        labels = kmeans.fit_predict(X=TF_IDF_matrix_)

        #
        # 4.-- Matriz de contingencia.
        #
        matrix = TF_IDF_matrix_.copy()
        matrix["*cluster*"] = labels
        matrix = matrix.groupby(by="*cluster*").sum()
        matrix.index = ["Cluster {:>2d}".format(i) for i in range(self.n_clusters)]
        self.contingency_table_ = matrix.transpose()

        #
        # 5.-- Memberships
        #
        memberships = None
        for cluster in self.contingency_table_.columns:
            grp = self.contingency_table_[cluster]
            grp = grp[grp > 0]
            grp = grp.sort_values(ascending=False)
            grp = grp.head(self.top_n)
            index = [(cluster, w) for w in grp.index]
            index = pd.MultiIndex.from_tuples(index, names=["Cluster", "Term"])
            grp = pd.DataFrame(grp.values, columns=["Values"], index=index)
            if memberships is None:
                memberships = grp
            else:
                memberships = pd.concat([memberships, grp])
        self.memberships_ = memberships

        #
        # 6.-- Top n for contingency table
        #
        self.contingency_table_ = cmn.sort_by_axis(
            data=self.contingency_table_, sort_by=self.top_by, ascending=False, axis=0
        )
        self.contingency_table_ = self.contingency_table_.head(self.top_n)

        #
        # 7.-- Nombres de los clusters.
        #
        self.cluster_names_ = {}
        for c in self.contingency_table_:
            m = self.contingency_table_.sort_values(by=c, ascending=False)
            self.cluster_names_[c] = m.index[0]

        #
        # 8.-- Tamaño de los clusters.
        #         Cantidad de documentos
        #         Cantidad de citas
        #
        W = self.data[[self.column, "Times_Cited"]]
        W = W.dropna()
        W.pop(self.column)
        W["Num_Documents"] = 1
        W["*cluster*"] = labels
        W = W.groupby("*cluster*").sum()
        self.num_documents_ = W["Num_Documents"].to_dict()
        self.times_cited_ = W["Times_Cited"].to_dict()

        #
        # 9.-- Correspondence Analysis
        #
        ca = CA()
        ca.fit(X=self.contingency_table_)
        self.cluster_ppal_coordinates_ = ca.principal_coordinates_cols_
        self.term_ppal_coordinates_ = ca.principal_coordinates_rows_

    def contingency_table(self):
        self.fit()
        return self.contingency_table_

    def memberships_table(self):
        self.fit()
        return self.memberships_

    def cluster_ppal_coordinates(self):
        self.fit()
        return self.cluster_ppal_coordinates_

    def term_ppal_coordinates(self):
        self.fit()
        return self.term_ppal_coordinates_

    def clusters_plot(self):
        self.fit()
        x = self.cluster_ppal_coordinates_[
            self.cluster_ppal_coordinates_.columns[self.x_axis]
        ]
        y = self.cluster_ppal_coordinates_[
            self.cluster_ppal_coordinates_.columns[self.y_axis]
        ]

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.cmap)

        node_sizes = self.num_documents_.values()
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        node_sizes = [
            500 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

        node_colors = self.times_cited_.values()
        max_colors = max(node_colors)
        min_colors = min(node_colors)
        node_colors = [
            cmap(0.2 + 0.80 * (i - min_colors) / (max_colors - min_colors))
            for i in node_colors
        ]

        ax.scatter(
            x,
            y,
            marker="o",
            s=node_sizes,
            alpha=1.0,
            c=node_colors,
            edgecolors="k",
            zorder=1,
        )

        ax.axhline(
            y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
        )
        ax.axvline(
            x=0, color="gray", linestyle="--", linewidth=1, zorder=-1,
        )

        dict_pos = {
            key: (x_, y_) for key, x_, y_ in zip(self.cluster_names_.keys(), x, y)
        }
        cmn.ax_text_node_labels(
            ax=ax, labels=self.cluster_names_, dict_pos=dict_pos, node_sizes=node_sizes
        )

        cmn.set_ax_splines_invisible(ax)
        cmn.ax_expand_limits(ax)
        ax.axis("off")

        fig.set_tight_layout(True)

        return fig

    def terms_plot(self):
        self.fit()
        x = self.term_ppal_coordinates_[
            self.term_ppal_coordinates_.columns[self.x_axis]
        ]
        y = self.term_ppal_coordinates_[
            self.term_ppal_coordinates_.columns[self.y_axis]
        ]

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.cmap)

        node_sizes = cmn.counters_to_node_sizes(self.term_ppal_coordinates_.index)
        node_colors = cmn.counters_to_node_colors(
            self.term_ppal_coordinates_.index, cmap
        )

        ax.scatter(
            x,
            y,
            marker="o",
            s=node_sizes,
            alpha=1.0,
            c=node_colors,
            edgecolors="k",
            zorder=1,
        )

        ax.axhline(
            y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
        )
        ax.axvline(
            x=0, color="gray", linestyle="--", linewidth=1, zorder=-1,
        )

        dict_pos = {
            key: (x_, y_)
            for key, x_, y_ in zip(self.term_ppal_coordinates_.index, x, y)
        }
        cmn.ax_text_node_labels(
            ax=ax,
            labels=self.term_ppal_coordinates_.index,
            dict_pos=dict_pos,
            node_sizes=node_sizes,
        )

        cmn.set_ax_splines_invisible(ax)
        cmn.ax_expand_limits(ax)
        ax.axis("off")

        fig.set_tight_layout(True)

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


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None):
        """Dashboard app"""

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.data = data
        self.app_title = "Thematic Analysis"
        self.menu_options = [
            "Contingency table",
            "Memberships table",
            "Cluster ppal coordinates",
            "Term ppal coordinates",
            "Clusters plot",
            "Terms plot",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            dash.top_n(n=101,),
            dash.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
            dash.dropdown(desc="Use IDF:", options=[True, False,],),
            dash.dropdown(desc="Smooth IDF:", options=[True, False,],),
            dash.dropdown(desc="Sublinear TF:", options=[True, False,],),
            dash.n_clusters(),
            dash.max_iter(),
            dash.cmap(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        self.panel_widgets[-4]["widget"].options = list(range(self.n_clusters))
        self.panel_widgets[-3]["widget"].options = list(range(self.n_clusters))

        for i in [-1, -2, -3, -4, -5]:
            self.panel_widgets[i]["widget"].disabled = self.menu in [
                "Contingency table",
                "Memberships table",
                "Cluster ppal coordinates",
                "Term ppal coordinates",
            ]


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()
