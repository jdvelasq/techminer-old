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
        ## Fuente:
        ##   https://tlab.it/en/allegati/help_en_online/mrepert.htm
        ##
        #
        # 1.-- Construye TF_matrix binaria
        #
        TF_matrix_ = TF_matrix(
            self.data, self.column, scheme="binary", min_occurrence=self.min_occurrence,
        )

        TF_matrix_ = cmn.add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 2.-- Construye TF-IDF y escala filas to longitud unitaria (norma euclidiana).
        #      En sklearn: norm='l2'
        #
        #      tf-idf = tf * (log(N / df) + 1)
        #
        TF_IDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm="l2",
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
            max_items=self.max_items,
        )

        #
        # 3.-- Clustering de las filas de TF_IDF_matrix_.
        #      En TLAB se usa bisecting k-means.
        #      Se implementa sklearn.cluster.KMeans
        #
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=int(self.random_state),
        )
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
        communities = pd.DataFrame(
            "", columns=range(self.n_clusters), index=range(self.top_n)
        )
        for i_cluster, cluster in enumerate(self.contingency_table_.columns):
            grp = self.contingency_table_[cluster]
            grp = grp[grp > 0]
            grp = cmn.sort_by_axis(
                data=grp, sort_by="Num Documents", ascending=False, axis=0
            )
            grp = grp.head(self.top_n)
            communities.at[0 : len(grp) - 1, i_cluster] = grp.index

        communities.columns = [
            "Theme {}".format(i) for i, _ in enumerate(self.contingency_table_.columns)
        ]
        self.memberships_ = communities

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
        #
        W = TF_IDF_matrix_.copy()
        W["*cluster*"] = labels
        W = W.groupby("*cluster*").count()

        #
        # 9.-- Correspondence Analysis
        #
        ca = CA()
        ca.fit(X=self.contingency_table_)
        self.cluster_ppal_coordinates_ = ca.principal_coordinates_cols_
        self.term_ppal_coordinates_ = ca.principal_coordinates_rows_

    def contingency_table(self):
        self.apply()
        return self.contingency_table_

    def memberships_table(self):
        self.apply()
        return self.memberships_

    def cluster_ppal_coordinates(self):
        self.apply()
        return self.cluster_ppal_coordinates_

    def term_ppal_coordinates(self):
        self.apply()
        return self.term_ppal_coordinates_

    def clusters_plot(self):
        self.apply()
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
        self.apply()
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
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

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
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Clustering (K-means)"),
            dash.n_clusters(m=3, n=51, i=1),
            dash.max_iter(),
            dash.random_state(),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            dash.top_n(n=101,),
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


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

