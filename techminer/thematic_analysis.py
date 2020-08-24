from techminer.core import add_counters_to_axis
from techminer.core import CA
from techminer.core import clustering
from techminer.core import DASH
from techminer.core import sort_by_axis
from techminer.core import TF_matrix, TFIDF_matrix
from techminer.plots import ax_text_node_labels
from techminer.plots import counters_to_node_colors
from techminer.plots import counters_to_node_sizes
from techminer.plots import expand_ax_limits
from techminer.plots import set_spines_invisible
import matplotlib
import matplotlib.pyplot as pyplot
import pandas as pd
import techminer.core.dashboard as dash

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

        ##
        ##  Construye TF_matrix binaria
        ##
        TF_matrix_ = TF_matrix(
            self.data, self.column, scheme="binary", min_occurrence=self.min_occurrence,
        )

        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ##  Construye TF-IDF y escala filas to longitud unitaria (norma euclidiana).
        ##      En sklearn: norm='l2'
        ##
        ##      tf-idf = tf * (log(N / df) + 1)
        ##
        TF_IDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm="l2",
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
            max_items=self.max_items,
        )

        ##
        ##  Clustering de las filas de TF_IDF_matrix_.
        ##      En TLAB se usa bisecting k-means.
        ##      Se implementa sklearn.cluster.KMeans
        ##
        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=TF_IDF_matrix_,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=None,
            name_prefix="Theme {}",
        )

        ##
        ##  Matriz de contingencia.
        ##
        matrix = TF_IDF_matrix_.copy()
        matrix["*cluster*"] = self.labels_
        matrix = matrix.groupby(by="*cluster*").sum()
        matrix.index = ["Theme {:>2d}".format(i) for i in range(self.n_clusters)]
        self.contingency_table_ = matrix.transpose()

        ##
        ##  Top n for contingency table
        ##
        self.contingency_table_ = sort_by_axis(
            data=self.contingency_table_, sort_by=self.top_by, ascending=False, axis=0
        )
        self.contingency_table_ = self.contingency_table_.head(self.top_n)

        ##
        ## Tamaño de los clusters
        ##
        W = TF_IDF_matrix_.copy()
        W["*cluster*"] = self.labels_
        W = W.groupby("*cluster*").count()[W.columns[0]].tolist()

        self.num_documents_ = W

        ##
        ##  Correspondence Analysis
        ##
        ca = CA()
        ca.fit(X=self.contingency_table_)
        self.cluster_ppal_coordinates_ = ca.principal_coordinates_cols_
        self.term_ppal_coordinates_ = ca.principal_coordinates_rows_

    def contingency_table(self):
        self.apply()
        return self.contingency_table_

    def cluster_members(self):
        self.apply()
        return self.cluster_members_

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

        node_sizes = self.num_documents_
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
        ax_text_node_labels(
            ax=ax, labels=self.cluster_names_, dict_pos=dict_pos, node_sizes=node_sizes
        )

        set_spines_invisible(ax)
        expand_ax_limits(ax)
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

        node_sizes = counters_to_node_sizes(self.term_ppal_coordinates_.index)
        node_colors = counters_to_node_colors(self.term_ppal_coordinates_.index, cmap)

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
        ax_text_node_labels(
            ax=ax,
            labels=self.term_ppal_coordinates_.index,
            dict_pos=dict_pos,
            node_sizes=node_sizes,
        )

        set_spines_invisible(ax)
        expand_ax_limits(ax)
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
            "Cluster members",
            "Contingency table",
            "Cluster ppal coordinates",
            "Term ppal coordinates",
            "Clusters plot",
            "Terms plot",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:",
                options=[z for z in sorted(COLUMNS) if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
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
                "Cluster members",
                "Cluster ppal coordinates",
                "Term ppal coordinates",
            ]


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def thematic_analysis(
    input_file="techminer.csv", limit_to=None, exclude=None, years_range=None
):
    return DASHapp(
        data=pd.read_csv(input_file),
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()

