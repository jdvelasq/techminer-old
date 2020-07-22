"""
Strategic and thematic maps
==================================================================================================



"""

## path_length = nx.shortest_path_length(self._graph)
## distances = pd.DataFrame(index=self._graph.nodes(), columns=self._graph.nodes())
## for row, data in path_length:
##     for col, dist in data.items():
##         distances.loc[row, col] = dist
## distances = distances.fillna(distances.max().max())
## return nx.kamada_kawai_layout(self._graph, dist=distances.to_dict())

import json

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

import techminer.by_term
import techminer.common as cmn
import techminer.graph as graph
import techminer.plots as plt
from techminer.document_term import TF_matrix
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50


class TABapp_:
    def __init__(self):
        self.output_ = widgets.Output()
        self.panel_ = []
        self.grid_ = []
        #
        self.view = None
        self.sort_by = None
        self.ascending = None
        self.cmap = None
        self.width = None
        self.height = None
        self.plot = None
        self.column = None
        self.norm = None
        self.use_idf = None
        self.smooth_idf = None
        self.top_n = None
        self.sublinear_tf = None
        self.by = None

    def run(self):
        return self.grid_

    def gui(self, **kwargs):

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def update(self, button):
        pass

    def create_grid(self):

        #  Grid size
        self.grid_ = GridspecLayout(max(14, len(self.panel_)), 6, height="650px")

        # Button control
        self.grid_[0, 0] = widgets.Button(
            description="Calculate",
            layout=Layout(width="91%", border="2px solid gray"),
        )
        self.grid_[0, 0].on_click(self.update)

        self.grid_[1:, 0] = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.Label(value=self.panel_[index]["desc"]),
                        self.panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                if self.panel_[index]["arg"] != "terms"
                else widgets.VBox(
                    [
                        widgets.Label(value=self.panel_[index]["desc"]),
                        self.panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self.panel_))
            ]
        )

        #  Output area
        self.grid_[:, 1:] = widgets.VBox(
            [self.output_], layout=Layout(height="650px", border="2px solid gray")
        )

        args = {control["arg"]: control["widget"] for control in self.panel_}
        with self.output_:
            display(widgets.interactive_output(self.gui, args,))


###################################################################################################
##
##  CALCULATIONS
##
###################################################################################################


class Strategic_map:
    def __init__(
        self,
        n_clusters=2,
        linkage="ward",
        top_by="Num Documents",
        top_n=10,
        normalization=None,
    ):

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.top_by = top_by
        self.top_n = top_n
        self.normalization = normalization

    def fit(self, data, column, limit_to=None, exclude=None):

        self.matrix_ = graph.co_occurrence_matrix(
            data=data,
            column=column,
            top_by=self.top_by,
            top_n=self.top_n,
            normalization=self.normalization,
            limit_to=limit_to,
            exclude=exclude,
        )

        clustering = AgglomerativeClustering(
            linkage=self.linkage, n_clusters=self.n_clusters
        )
        clustering.fit(1 - self.matrix_)
        cluster_dict = {
            key: value for key, value in zip(self.matrix_.columns, clustering.labels_)
        }

        map = pd.DataFrame(
            {"cluster": list(range(self.n_clusters))},
            index=list(range(self.n_clusters)),
        )
        map["density"] = 0.0
        map["centrality"] = 0.0
        map["name"] = ""
        map["n_members"] = 0
        map["members"] = [[]] * len(map)

        #
        # Members of cluster
        #
        for t in self.matrix_.columns:
            map.at[cluster_dict[t], "members"] = map.loc[cluster_dict[t], "members"] + [
                t
            ]

        #
        # Name of cluster
        #
        for i_cluster, words in enumerate(map["members"]):
            cluster_name = None
            cluster_freq = None
            map.at[i_cluster, "n_members"] = len(words)
            for word in words:
                freq = int(word.split(" ")[-1].split(":")[0])
                if cluster_freq is None or freq > cluster_freq:
                    cluster_name = word
                    cluster_freq = freq
            map.at[i_cluster, "name"] = cluster_name

        for i_cluster in range(len(map)):
            map.at[i_cluster, "members"] = ";".join(map.loc[i_cluster, "members"])

        #
        # density
        #
        for i_cluster in range(self.n_clusters):
            Z = self.matrix_[
                [t for t in self.matrix_.columns if cluster_dict[t] == i_cluster]
            ]
            I = Z.copy()
            I = I.loc[[t for t in I.index if cluster_dict[t] == i_cluster], :]
            if len(I) == 1:
                map.at[i_cluster, "density"] = 0
            else:
                density = []
                for i_column in range(len(I.columns) - 1):
                    for i_index in range(i_column + 1, len(I.index)):
                        density.append(I.loc[I.index[i_index], I.columns[i_column]])

                map.at[i_cluster, "density"] = sum(density) / len(density)

        #
        # centratity
        #
        for i_cluster in range(self.n_clusters):
            Z = self.matrix_[
                [t for t in self.matrix_.columns if cluster_dict[t] == i_cluster]
            ]
            I = Z.copy()
            I = I.loc[[t for t in I.index if cluster_dict[t] != i_cluster], :]
            map.at[i_cluster, "centrality"] = I.sum().sum()

        self.map_ = map

        ## Memberships
        memberships = {}
        for i_cluster in range(self.n_clusters):
            memberships[map.name[i_cluster]] = map.members[i_cluster].split(";")

        self.memberships_ = memberships

    def plot(self, cmap="Greys", figsize=(6, 6)):
        pass


###################################################################################################
##
##  APP
##
###################################################################################################


def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Strategic Map",
        tab_titles=["Strategic Map"],
        tab_widgets=[TABapp0(data, limit_to=limit_to, exclude=exclude).run(),],
        tab=tab,
    )


###################################################################################################
##
##  TAB app 0 --- Strategic Map
##
###################################################################################################


MAP_COLUMNS = [
    "Author_Keywords",
    "Author_Keywords_CL",
    "Index_Keywords",
    "Index_Keywords_CL",
    "Abstract_words",
    "Title_words",
    "Abstract_words_CL",
    "Title_words_CL",
]


class TABapp0(gui.TABapp_):
    def __init__(self, data, limit_to, exclude):

        super(TABapp0, self).__init__()

        self.data_ = data
        self.limit_to_ = limit_to
        self.exclude_ = exclude

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.panel_ = [
            dash.dropdown(desc="View:", options=["Table", "Membership", "Plot"],),
            dash.dropdown(
                desc="Column:", options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.dropdown(desc="Top by:", options=["Num Documents", "Times Cited"],),
            dash.top_n(m=10, n=1001, i=10),
            dash.normalization(),
            dash.n_clusters(),
            dash.linkage(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def gui(self, **kwargs):

        super().gui(**kwargs)

        self.panel_[-3]["widget"].disabled = False if self.view == "Plot" else True
        self.panel_[-2]["widget"].disabled = False if self.view == "Plot" else True
        self.panel_[-1]["widget"].disabled = False if self.view == "Plot" else True

    def update(self, button):
        """ 
        """

        self.output_.clear_output()
        with self.output_:
            display(gui.processing())

        matrix_ = graph.co_occurrence_matrix(
            data=self.data_,
            column=self.column,
            top_by=self.top_by,
            top_n=self.top_n,
            normalization=self.normalization,
            limit_to=self.limit_to_,
            exclude=self.exclude_,
        )

        self.strategic_map = Strategic_map(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            top_by=self.top_by,
            top_n=self.top_n,
            normalization=self.normalization,
        )

        self.strategic_map.fit(
            data=self.data_,
            column=self.column,
            limit_to=self.limit_to_,
            exclude=self.exclude_,
        )

        self.output_.clear_output()
        with self.output_:
            if self.view == "Table":
                display(self.strategic_map.map_)

            if self.view == "Membership":
                print(
                    json.dumps(
                        self.strategic_map.memberships_, indent=4, sort_keys=True
                    )
                )

            if self.view == "Plot":
                display(
                    self.strategic_map.plot(
                        cmap=self.cmap, figsize=(self.width, self.height)
                    )
                )


##################################################################################################
