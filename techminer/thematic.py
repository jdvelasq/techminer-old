import json

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import techminer.by_term as by_term
import techminer.common as cmn
import techminer.gui as gui
from IPython.display import clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from techminer.document_term import TF_matrix, TFIDF_matrix

from techminer.correspondence import CA

import techminer.plots as plt

from sklearn.cluster import KMeans
from techminer.dashboard import DASH


###############################################################################
##
##  Model
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
        self.sort_by = None
        self.column = None
        self.top_by = None
        self.top_n = None
        self.cmap = None
        self.height = None
        self.width = None
        self.smooth_idf = None
        self.use_idf = None
        self.sublinear_tf = None
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
            gui.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
            gui.top_n(n=101,),
            gui.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
            gui.dropdown(desc="Use IDF:", options=[True, False,],),
            gui.dropdown(desc="Smooth IDF:", options=[True, False,],),
            gui.dropdown(desc="Sublinear TF:", options=[True, False,],),
            gui.n_clusters(),
            gui.max_iter(),
            gui.cmap(),
            gui.x_axis(),
            gui.y_axis(),
            gui.fig_width(),
            gui.fig_height(),
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


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #


# ################################################################################################
# ##
# ##  CALCULATIONS
# ##
# ################################################################################################


# class Thematic_analysis:
#     def __init__(
#         self,
#         norm,
#         use_idf,
#         smooth_idf,
#         sublinear_tf,
#         n_clusters,
#         max_iter,
#         top_by,
#         top_n,
#     ):

#         self.norm = norm
#         self.use_idf = use_idf
#         self.smooth_idf = smooth_idf
#         self.sublinear_tf = sublinear_tf
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.top_by = top_by
#         self.top_n = top_n

#     def fit(self, data, column, limit_to=None, exclude=None):

#         ##
#         ## Fuente:
#         ##   https://tlab.it/en/allegati/help_en_online/mrepert.htm
#         ##

#         #
#         # 1.-- Construye TF_matrix binaria
#         #
#         TF_matrix_ = TF_matrix(data, column, scheme="binary")
#         TF_matrix_ = cmn.limit_to_exclude(
#             data=TF_matrix_, axis=1, column=column, limit_to=limit_to, exclude=exclude,
#         )
#         TF_matrix_ = cmn.add_counters_to_axis(
#             X=TF_matrix_, axis=1, data=data, column=column
#         )
#         TF_matrix_ = cmn.sort_by_axis(
#             data=TF_matrix_, sort_by=self.top_by, ascending=False, axis=1
#         )
#         TF_matrix_ = TF_matrix_[TF_matrix_.columns[: self.top_n]]

#         #
#         # 2.-- Construye TF-IDF y escala filas to longitud unitaria (norma euclidiana).
#         #      En sklearn: norm='l2'
#         #
#         #      En TLAB se usa tf * log( N / df).
#         #      Para sklearn:
#         #        * use_idf = False
#         #        * sublinear_tf = False
#         #
#         TF_IDF_matrix_ = TFIDF_matrix(
#             TF_matrix=TF_matrix_,
#             norm=self.norm,
#             use_idf=self.use_idf,
#             smooth_idf=self.smooth_idf,
#             sublinear_tf=self.sublinear_tf,
#         )

#         #
#         # 3.-- Clustering de las filas de TF_IDF_matrix_.
#         #      En TLAB se usa bisecting k-means.
#         #      Se implementa sklearn.cluster.KMeans
#         #
#         kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
#         labels = kmeans.fit_predict(X=TF_IDF_matrix_)

#         #
#         # 4.-- Matriz de contingencia.
#         #
#         matrix = TF_IDF_matrix_.copy()
#         matrix["*cluster*"] = labels
#         matrix = matrix.groupby(by="*cluster*").sum()
#         matrix.index = ["Cluster {:>2d}".format(i) for i in range(self.n_clusters)]
#         self.contingency_table_ = matrix.transpose()

#         #
#         # 5.-- Nombres de los clusters.
#         #
#         self.cluster_names_ = {}
#         for c in self.contingency_table_:
#             m = self.contingency_table_.sort_values(by=c, ascending=False)
#             self.cluster_names_[c] = m.index[0]

#         #
#         # 6.-- Tamaño de los clusters.
#         #         Cantidad de documentos
#         #         Cantidad de citas
#         #
#         W = data[[column, "Times_Cited"]]
#         W = W.dropna()
#         W.pop(column)
#         W["Num_Documents"] = 1
#         W["*cluster*"] = labels
#         W = W.groupby("*cluster*").sum()
#         self.num_documents_ = W["Num_Documents"].to_dict()
#         self.times_cited_ = W["Times_Cited"].to_dict()

#         #
#         # 7.-- Correspondence Analysis
#         #
#         ca = CA()
#         ca.fit(X=self.contingency_table_)
#         self.cluster_ppal_coordinates_ = ca.principal_coordinates_cols_
#         self.terms_ppal_coordinates_ = ca.principal_coordinates_rows_


################################################################################################
##
##  TAB app 0 --- Thematic Analysis
##
################################################################################################


# class TABapp0(gui.TABapp_):
#     def __init__(self, data, limit_to, exclude):

#         super(TABapp0, self).__init__()

#         self.data_ = data
#         self.limit_to_ = limit_to
#         self.exclude_ = exclude

#         COLUMNS = [
#             "Authors",
#             "Countries",
#             "Institutions",
#             "Author_Keywords",
#             "Index_Keywords",
#             "Abstract_words_CL",
#             "Abstract_words",
#             "Title_words_CL",
#             "Title_words",
#             "Affiliations",
#             "Author_Keywords_CL",
#             "Index_Keywords_CL",
#         ]

#         self.panel_ = [
#             gui.dropdown(
#                 desc="View:",
#                 options=[
#                     "Contigency table",
#                     "Membership",
#                     "Cluster ppal coordinates",
#                     "Plot clusters",
#                     "Plot terms",
#                 ],
#             ),
#             gui.dropdown(
#                 desc="Column:", options=[z for z in COLUMNS if z in data.columns],
#             ),
#             gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
#             gui.top_n(n=101,),
#             gui.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
#             gui.dropdown(desc="Use IDF:", options=[True, False,],),
#             gui.dropdown(desc="Smooth IDF:", options=[True, False,],),
#             gui.dropdown(desc="Sublinear TF:", options=[True, False,],),
#             gui.n_clusters(),
#             gui.max_iter(),
#             gui.cmap(),
#             gui.x_axis(),
#             gui.y_axis(),
#             gui.fig_width(),
#             gui.fig_height(),
#         ]
#         super().create_grid()

#     def gui(self, **kwargs):

#         super().gui(**kwargs)
#         self.panel_[-4]["widget"].options = list(range(self.n_clusters))
#         self.panel_[-3]["widget"].options = list(range(self.n_clusters))

#         for i in [-1, -2, -3, -4, -5]:
#             self.panel_[i]["widget"].disabled = kwargs["view"] in [
#                 "Contigency table",
#                 "Membership",
#                 "Cluster ppal coordinates",
#             ]

#         #  self.panel_[-1]["widget"].disabled = kwargs["view"] == "Contigency table"

#     def update(self, button):

#         self.output_.clear_output()
#         with self.output_:
#             display(gui.processing())

#         thematic = Thematic_analysis(
#             norm=self.norm,
#             use_idf=self.use_idf,
#             smooth_idf=self.smooth_idf,
#             sublinear_tf=self.sublinear_tf,
#             n_clusters=self.n_clusters,
#             max_iter=self.max_iter,
#             top_by=self.top_by,
#             top_n=self.top_n,
#         )

#         thematic.fit(
#             data=self.data_,
#             column=self.column,
#             limit_to=self.limit_to_,
#             exclude=self.exclude_,
#         )

#         self.output_.clear_output()
#         with self.output_:

#             if self.view == "Contigency table":
#                 display(thematic.contingency_table_)

#             if self.view == "Membership":
#                 print("=" * 55)
#                 print("    {:>40s}   {:>6s}".format("Term", "TF*IDF"))
#                 for k in thematic.cluster_names_.keys():
#                     print(("-" * 55))
#                     print(k, ": ", thematic.cluster_names_[k], sep="")
#                     print("")
#                     m = thematic.contingency_table_.sort_values(k, ascending=False)
#                     m = m[k]
#                     m = m[m > 0.0]
#                     for i in m.index:
#                         print("    {:>40s}   {:>6.2f}".format(i, m[i]))
#                     print("")
#                     print("")

#             if self.view == "Cluster ppal coordinates":
#                 display(thematic.cluster_ppal_coordinates_)

#             if self.view == "Plot clusters":
#                 display(
#                     thematic.plot_clusters(
#                         x_axis=int(self.x_axis),
#                         y_axis=int(self.y_axis),
#                         cmap=self.cmap,
#                         figsize=(self.width, self.height),
#                     )
#                 )

#             if self.view == "Plot terms":
#                 display(
#                     thematic.plot_terms(
#                         x_axis=int(self.x_axis),
#                         y_axis=int(self.y_axis),
#                         cmap=self.cmap,
#                         figsize=(self.width, self.height),
#                     )
#                 )


################################################################################################


# def _get_fmt(summ):
#     n_Num_Documents = int(np.log10(summ["Num_Documents"].max())) + 1
#     n_Times_Cited = int(np.log10(summ["Times_Cited"].max())) + 1
#     return "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"


# def thematc_analysis(
#     data,
#     column,
#     top_by,
#     top_n,
#     n_clusters=2,
#     linkage="ward",
#     output=0,
#     n_components=2,
#     x_axis=0,
#     y_axis=0,
#     cmap="Greys",
#     figsize=(6, 6),
#     limit_to=None,
#     exclude=None,
# ):

#     dtm = TF_matrix(data, column)

#     summ = by_term.analytics(data, column)
#     fmt = _get_fmt(summ)
#     new_names = {
#         key: fmt.format(key, nd, tc)
#         for key, nd, tc in zip(summ.index, summ.Num_Documents, summ.Times_Cited)
#     }

#     #
#     # Select top N terms
#     #
#     if isinstance(top_by, str):
#         top_by = top_by.replace(" ", "_")
#         top_by = {"Num_Documents": 0, "Times_Cited": 1,}[top_by]

#     if top_by == 0:
#         summ = summ.sort_values(
#             ["Num_Documents", "Times_Cited"], ascending=[False, False],
#         )

#     if top_by == 1:
#         summ = summ.sort_values(
#             ["Times_Cited", "Num_Documents"], ascending=[False, False, True],
#         )

#     if isinstance(limit_to, dict):
#         if column in limit_to.keys():
#             limit_to = limit_to[column]
#         else:
#             limit_to = None

#     if limit_to is not None:
#         summ = summ[summ.index.map(lambda w: w in limit_to)]

#     if isinstance(exclude, dict):
#         if column in exclude.keys():
#             exclude = exclude[column]
#         else:
#             exclude = None

#     if exclude is not None:
#         summ = summ[summ.index.map(lambda w: w not in exclude)]

#     top_terms = summ.head(top_n).index.tolist()

#     dtm = dtm[[t for t in dtm.columns if t in top_terms]]

#     dtm.columns = [new_names[t] for t in dtm.columns]

#     #
#     # processing
#     #
#     m = dtm.sum(axis=1)
#     m = m[m > 0]
#     dtm = dtm.loc[m.index, :]

#     dtm = dtm.transpose()
#     dtm = dtm.applymap(lambda w: 1 if w > 0 else 0)
#     ndocs = dtm.sum(axis=0)
#     N = len(dtm)
#     for col in dtm.columns:
#         dtm[col] = dtm[col].map(lambda w: w * np.log(N / ndocs[col]))
#     for index in dtm.index:
#         s = dtm.loc[index, :].tolist()
#         n = np.sqrt(sum([u ** 2 for u in s]))
#         dtm.at[index, :] = dtm.loc[index, :] / n

#     clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
#     clustering.fit(dtm)
#     cluster_dict = {key: value for key, value in zip(dtm.index, clustering.labels_)}

#     map = pd.DataFrame(
#         {"cluster": list(range(n_clusters))}, index=list(range(n_clusters))
#     )
#     map["name"] = ""
#     map["n_members"] = 0
#     map["members"] = [[]] * len(map)

#     #
#     # Members of cluster
#     #
#     for t in dtm.index:
#         map.at[cluster_dict[t], "members"] = map.loc[cluster_dict[t], "members"] + [t]

#     #
#     # Name of cluster
#     #
#     for i_cluster, words in enumerate(map["members"]):
#         cluster_name = None
#         cluster_freq = None
#         map.at[i_cluster, "n_members"] = len(words)
#         for word in words:
#             freq = int(word.split(" ")[-1].split(":")[0])
#             if cluster_freq is None or freq > cluster_freq:
#                 cluster_name = word
#                 cluster_freq = freq
#         map.at[i_cluster, "name"] = cluster_name

#     for i_cluster in range(len(map)):
#         map.at[i_cluster, "members"] = ";".join(map.loc[i_cluster, "members"])

#     if output == 0:
#         text = {}
#         for i_cluster in range(n_clusters):
#             text[map.name[i_cluster]] = map.members[i_cluster].split(";")
#         return json.dumps(text, indent=4, sort_keys=True)

#     if output == 1:

#         #
#         # Representation using multidimensinal scaling
#         #
#         embedding = MDS(n_components=n_components)
#         dtm_transformed = embedding.fit_transform(dtm,)

#         matplotlib.rc("font", size=11)
#         cmap = pyplot.cm.get_cmap(cmap)
#         fig = pyplot.Figure(figsize=figsize)
#         ax = fig.subplots()

#         node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in map.name]
#         max_size = max(node_sizes)
#         min_size = min(node_sizes)
#         node_sizes = [
#             600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
#         ]

#         node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in map.name]
#         max_citations = max(node_colors)
#         min_citations = min(node_colors)
#         node_colors = [
#             cmap(0.2 + 0.80 * (i - min_citations) / (max_citations - min_citations))
#             for i in node_colors
#         ]

#         x_clusters = []
#         y_clusters = []
#         for i_cluster in range(n_clusters):
#             x = dtm_transformed[clustering.labels_ == i_cluster, x_axis].mean()
#             y = dtm_transformed[clustering.labels_ == i_cluster, y_axis].mean()
#             x_clusters.append(x)
#             y_clusters.append(y)

#         ax.scatter(
#             x_clusters,
#             y_clusters,
#             s=node_sizes,
#             linewidths=1,
#             edgecolors="k",
#             c=node_colors,
#         )

#         common.ax_expand_limits(ax)

#         pos = {
#             key: (x_clusters[idx], y_clusters[idx]) for idx, key in enumerate(map.name)
#         }
#         common.ax_text_node_labels(
#             ax=ax, labels=map.name, dict_pos=pos, node_sizes=node_sizes
#         )

#         ax.axhline(
#             y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
#         )
#         ax.axvline(
#             x=0, color="gray", linestyle="--", linewidth=1, zorder=-1,
#         )

#         common.set_ax_splines_invisible(ax)
#         ax.axis("off")

#         fig.set_tight_layout(True)

#         return fig

#     return None
