import textwrap
import re
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from pyvis.network import Network
import matplotlib
import networkx as nx

import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    Network,
    TF_matrix,
    add_counters_to_axis,
    corpus_filter,
    limit_to_exclude,
    normalize_network,
    sort_by_axis,
    explode,
)
from techminer.core.params import EXCLUDE_COLS
from techminer.core import cluster_table_to_list
from techminer.plots import ChordDiagram
from techminer.plots import bubble_plot as bubble_plot_
from techminer.plots import counters_to_node_colors, counters_to_node_sizes
from techminer.plots import heatmap as heatmap_
from techminer.plots import (
    ax_text_node_labels,
    expand_ax_limits,
)
from techminer.plots import set_spines_invisible

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
        top_n,
        limit_to,
        exclude,
        years_range,
        clusters=None,
        cluster=None,
    ):
        #
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
        self.top_n = top_n
        self.clusters = clusters
        self.cluster = cluster

        self.cmap = None
        self.column = None
        self.height = None
        self.keyword = None
        self.max_items = None
        self.min_occurrence = None
        self.normalization = None
        self.width = None

    def radial_diagram(self):

        ##
        ##  Computes TF_matrix with occurrence >= min_occurrence
        ##
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occurrence,
        )

        ##
        ##  Limit to/Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ##  Adds counters to axis
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )
        TF_matrix_ = sort_by_axis(
            data=TF_matrix_, sort_by="Num_Documents", ascending=False, axis=1
        )

        ##
        ##  Select max_items
        ##
        TF_matrix_ = TF_matrix_[TF_matrix_.columns[: self.max_items]]
        if len(TF_matrix_.columns) > self.max_items:
            top_items = TF_matrix_.sum(axis=0)
            top_items = top_items.sort_values(ascending=False)
            top_items = top_items.head(self.max_items)
            TF_matrix_ = TF_matrix_.loc[:, top_items.index]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        ##
        ## Remove counters from axes
        ##
        # TF_matrix_.columns = [" ".join(w.split(" ")[:-1]) for w in TF_matrix_.columns]

        ##
        ##  Co-occurrence matrix and association index
        ##
        X = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        X = pd.DataFrame(X, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        X = normalize_network(X, self.normalization)

        ##
        ## Selected Keyword
        ##
        keyword = [
            w
            for w in TF_matrix_.columns.tolist()
            if (" ".join(w.split(" ")[:-1]).lower() == self.keyword)
        ]
        if len(keyword) > 0:
            keyword = keyword[0]
        else:
            return widgets.HTML("<pre>No associations for the selected keyword")

        ##
        ## Network plot
        ##
        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.cmap)

        ##
        ## Selects the column with values > 0
        ##
        X = X[[keyword]]
        X = X[X.index != keyword]
        X = X[X[keyword] > 0]

        nodes = X.index.tolist() + [keyword]
        node_sizes = counters_to_node_sizes(nodes)
        node_colors = counters_to_node_colors(x=nodes, cmap=lambda w: w)
        node_colors = [cmap(t) for t in node_colors]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i, w in zip(X.index, X[keyword]):
            G.add_edge(i, keyword, width=w)

        ##
        ## Layout
        ##
        pos = nx.kamada_kawai_layout(G)

        ##
        ## Draw network edges
        ##
        for e in G.edges.data():
            a, b, dict_ = e
            edge = [(a, b)]
            width = 0.5 + 2.0 * dict_["width"]
            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                edgelist=edge,
                width=width,
                edge_color="k",
                node_size=1,
                alpha=0.5,
            )

        ##
        ## Draw network nodes
        ##
        for i_node, node in enumerate(G.nodes.data()):
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=[node[0]],
                node_size=node_sizes[i_node],
                node_color=node_colors[i_node],
                node_shape="o",
                edgecolors="k",
                linewidths=1,
                alpha=0.8,
            )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for i_node, label in enumerate(nodes):
            x_point, y_point = pos[label]
            ax.text(
                x_point
                + 0.01 * (xlim[1] - xlim[0])
                + 0.001 * node_sizes[i_node] / 300 * (xlim[1] - xlim[0]),
                y_point
                - 0.01 * (ylim[1] - ylim[0])
                - 0.001 * node_sizes[i_node] / 300 * (ylim[1] - ylim[0]),
                s=label,
                fontsize=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                horizontalalignment="left",
                verticalalignment="top",
            )

        fig.set_tight_layout(True)
        expand_ax_limits(ax)
        set_spines_invisible(ax)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig

    def concordances(self):

        data = self.data.copy()
        data = data[
            ["Authors", "Historiograph_ID", "Abstract", "Global_Citations"]
        ].dropna()
        data["Authors"] = data.Authors.map(lambda w: w.replace(";", ", "))
        data["REF"] = (
            data.Authors
            + ". "
            + data.Historiograph_ID
            + ". Times Cited: "
            + data.Global_Citations.map(str)
        )
        data = data[["REF", "Abstract", "Global_Citations"]]
        data["Abstract"] = data.Abstract.map(lambda w: w.split(". "))
        data = data.explode("Abstract")
        data = data[data.Abstract.map(lambda w: self.keyword.lower() in w.lower())]
        data = data.groupby(["REF", "Global_Citations"], as_index=False).agg(
            {"Abstract": list}
        )
        data["Abstract"] = data.Abstract.map(lambda w: ". <br><br>".join(w))
        data["Abstract"] = data.Abstract.map(lambda w: w + ".")
        data = data.sort_values(["Global_Citations", "REF"], ascending=[False, True])
        data = data.head(50)
        #  pattern = re.compile(r"\b" + self.keyword + r"\b", re.IGNORECASE)
        pattern = re.compile(self.keyword, re.IGNORECASE)
        data["Abstract"] = data.Abstract.map(
            lambda w: pattern.sub("<b>" + self.keyword.upper() + "</b>", w)
        )

        HTML = ""
        for ref, phrase in zip(data.REF, data.Abstract):
            HTML += "=" * 80 + "<br>"
            HTML += ref + "<br><br>"
            phrases = textwrap.wrap(phrase, 80)
            for line in phrases:
                HTML += line + "<br>"
            HTML += "<br>"

        return widgets.HTML("<pre>" + HTML + "</pre>")


###############################################################################
##
##  DASHBOARD
##
###############################################################################

COLUMNS = [
    "Author_Keywords",
    "Author_Keywords_CL",
    "Index_Keywords",
    "Index_Keywords_CL",
    "Keywords_CL",
    "Title_Keywords",
    "Title_Keywords_CL" "Abstract_Keywords",
    "Abstract_Keywords_CL",
]


class DASHapp(DASH, Model):
    def __init__(
        self,
        data,
        top_n=50,
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
            top_n=top_n,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
            clusters=clusters,
            cluster=cluster,
        )
        DASH.__init__(self)

        self.app_title = "Keywords Association"
        self.menu_options = [
            "Concordances",
            "Radial Diagram",
        ]

        self.panel_widgets = [
            dash.dropdown(
                desc="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.dropdown(
                desc="Keyword:",
                options=[],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.normalization(include_none=False),
            dash.separator(text="Visualization"),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.clusters is not None and self.column == self.clusters[0]:
            #
            # Populates value control with the terms in the cluster
            #
            self.panel_widgets[1]["widget"].options = sorted(
                self.clusters[1][self.cluster]
            )

        elif self.top_n is not None:
            #
            # Populate value control with top_n terms
            #
            y = self.data.copy()
            y["Num_Documents"] = 1
            y = explode(
                y[
                    [
                        self.column,
                        "Num_Documents",
                        "Global_Citations",
                        "ID",
                    ]
                ],
                self.column,
            )
            y = y.groupby(self.column, as_index=True).agg(
                {
                    "Num_Documents": np.sum,
                    "Global_Citations": np.sum,
                }
            )
            y["Global_Citations"] = y["Global_Citations"].map(lambda w: int(w))
            top_terms_freq = set(
                y.sort_values("Num_Documents", ascending=False).head(self.top_n).index
            )
            top_terms_cited_by = set(
                y.sort_values("Global_Citations", ascending=False)
                .head(self.top_n)
                .index
            )
            top_terms = sorted(top_terms_freq | top_terms_cited_by)
            self.panel_widgets[1]["widget"].options = top_terms

        else:

            #
            # Populate Keywords with all terms
            #
            x = explode(self.data, self.column)
            all_terms = pd.Series(x[self.column].unique())
            all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
            all_terms = all_terms.sort_values()
            self.panel_widgets[1]["widget"].options = all_terms


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def keywords_association(
    input_file="techminer.csv",
    top_n=50,
    limit_to=None,
    exclude=None,
    years_range=None,
    clusters=None,
    cluster=None,
):
    return DASHapp(
        data=pd.read_csv(input_file),
        top_n=top_n,
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
        clusters=clusters,
        cluster=cluster,
    ).run()
