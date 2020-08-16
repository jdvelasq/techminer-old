"""
Factor analysis
==================================================================================================



"""
import matplotlib.pyplot as pyplot
import networkx as nx
from numpy.lib.index_tricks import RClass
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from techminer.clustering import clustering
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
import matplotlib.pyplot as pyplot

import techminer.common as cmn
import techminer.dashboard as dash
from techminer.dashboard import DASH
from techminer.tfidf import TF_matrix, TFIDF_matrix


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

        # TLAB suggestion
        self.n_components = 20

    def apply(self):
        #
        X = self.data.copy()

        #
        # 1.-- TF matrix
        #
        M = TF_matrix(
            data=X, column=self.column, scheme=None, min_occurrence=self.min_occurrence,
        )

        #
        # 2.-- Limit to / Exclude
        #
        M = cmn.limit_to_exclude(
            data=M,
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
        if self.apply_to == "TF*IDF matrix":
            M = TFIDF_matrix(
                TF_matrix=M,
                norm=None,
                use_idf=False,
                smooth_idf=False,
                sublinear_tf=False,
                max_items=self.max_items,
            )

        #
        # 4.-- Add counters to axes
        #
        M = cmn.add_counters_to_axis(X=M, axis=1, data=self.data, column=self.column)

        #
        # 5.-- Transpose
        #
        M = M.transpose()

        #
        # 6.-- Factor decomposition
        #
        model = {
            "Factor Analysis": FactorAnalysis,
            "PCA": PCA,
            "Fast ICA": FastICA,
            "SVD": TruncatedSVD,
            "MDS": MDS,
        }[self.method](
            n_components=self.n_components, random_state=int(self.random_state)
        )

        if self.method == "MDS":
            R = model.fit_transform(X=M.values)
        else:
            R = model.fit_transform(X=M.values)
        R = pd.DataFrame(
            R,
            columns=["Dim-{:>02d}".format(i) for i in range(self.n_components)],
            index=M.index,
        )

        #
        # 7.-- limit to/exclude terms
        #
        R = cmn.limit_to_exclude(
            data=R,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        #
        # 8.-- Clustering
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

        ##
        R["Cluster"] = self.labels_

        #
        # 12.-- Results
        #
        self.X_ = R

    def cluster_members(self):
        self.apply()
        return self.cluster_members_

    def cluster_plot(self):
        ##
        self.apply()
        ##

        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()

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

        x = self.centers_["Dim-{:>02d}".format(self.x_axis)]
        y = self.centers_["Dim-{:>02d}".format(self.y_axis)]
        names = self.centers_["Name"]
        node_sizes = cmn.counters_to_node_sizes(names)

        # node_colors = cmn.counters_to_node_colors(names, cmap)
        # edge_colors = cmn.counters_to_edgecolors(names, cmap)

        from cycler import cycler

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

        pos = {term: (x[idx], y[idx]) for idx, term in enumerate(self.centers_.Name)}
        cmn.ax_text_node_labels(
            ax=ax, labels=self.centers_.Name, dict_pos=pos, node_sizes=node_sizes
        )

        cmn.ax_expand_limits(ax)
        cmn.set_ax_splines_invisible(ax)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.set_axis_off()

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

###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        #
        self.app_title = "Latent Semantic Analysis"
        self.menu_options = [
            "Cluster members",
            "Cluster plot",
        ]
        #
        self.panel_widgets = [
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Decomposition"),
            dash.dropdown(desc="Apply to:", options=["TF matrix", "TF*IDF matrix",],),
            dash.dropdown(
                desc="Method:",
                options=["Factor Analysis", "PCA", "Fast ICA", "SVD", "MDS"],
            ),
            ## dash.n_components(), # n_components = 20
            dash.random_state(),
            dash.separator(text="Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.separator(text="Visualization"),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
            dash.top_n(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Memberships":
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu == "Cluster plot":
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.clustering_method in ["Affinity Propagation"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            #  self.set_enabled("Random State:")

        if self.clustering_method in ["Agglomerative Clustering"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            # self.set_disabled("Random State:")

        if self.clustering_method in ["Birch"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            # self.set_disabled("Random State:")

        if self.clustering_method in ["DBSCAN"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            # self.set_disabled("Random State:")

        if self.clustering_method in ["Feature Agglomeration"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            #  self.set_disabled("Random State:")

        if self.clustering_method in ["KMeans"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            #  self.set_disabled("Random State:")

        if self.clustering_method in ["Mean Shift"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            #  self.set_disabled("Random State:")

        self.set_options(name="X-axis:", options=list(range(self.n_components)))
        self.set_options(name="Y-axis:", options=list(range(self.n_components)))


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

