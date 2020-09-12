import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import pandas as pd
from cdlib import algorithms
from pyvis.network import Network as Network_

from techminer.core.sort_axis import sort_axis
from techminer.plots import (
    counters_to_node_sizes,
    expand_ax_limits,
    set_spines_invisible,
)

cluster_colors = [
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

cluster_colors += cluster_colors + cluster_colors


class Network:
    def __init__(self, X, top_by, n_labels, clustering):

        X = X.copy()

        ##
        ##  Network generation
        ##
        G = nx.Graph()

        ##
        ##  Top terms for labels
        ##
        X = sort_axis(
            data=X,
            num_documents=(top_by == "Num Documents"),
            axis=1,
            ascending=False,
        )
        self.top_terms_ = X.columns.tolist()[:n_labels]

        ##
        ##  Add nodes to the network
        ##
        terms = X.columns.tolist()
        G.add_nodes_from(terms)

        ##
        ##  Adds size property to nodes
        ##
        node_sizes = counters_to_node_sizes(terms)
        for term, size in zip(terms, node_sizes):
            G.nodes[term]["size"] = size

        ##
        ##  Add edges to the network
        ##
        m = X.stack().to_frame().reset_index()
        m = m[m.level_0 < m.level_1]
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.001]
        m = m.reset_index(drop=True)
        for idx in range(len(m)):
            G.add_edge(
                m.from_[idx],
                m.to_[idx],
                width=m.link_[idx],
                color="lightgray",
                physics=False,
            )

        ##
        ##  Network clustering
        ##
        R = {
            "Label propagation": algorithms.label_propagation,
            "Leiden": algorithms.leiden,
            "Louvain": algorithms.louvain,
            "Walktrap": algorithms.walktrap,
        }[clustering](G).communities

        for i_community, community in enumerate(R):
            for item in community:
                G.nodes[item]["group"] = i_community

        ##
        ##  Cluster members
        ##
        n_communities = len(R)
        max_len = max([len(r) for r in R])
        communities = pd.DataFrame(
            "", columns=range(n_communities), index=range(max_len)
        )
        for i_community in range(n_communities):
            community = R[i_community]
            community = sorted(
                community, key=(lambda w: w.split(" ")[-1]), reverse=True
            )
            communities.at[0 : len(community) - 1, i_community] = community
        communities = communities.head(n_labels)
        communities.columns = ["Cluster {}".format(i) for i in range(n_communities)]

        self.cluster_members_ = communities

        ##
        ##  Saves the graph
        ##
        self.G_ = G

    def networkx_plot(self, layout, iterations, figsize):

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()

        if layout == "Spring":
            pos = nx.spring_layout(self.G_, iterations=iterations)
        else:
            pos = {
                "Circular": nx.circular_layout,
                "Kamada Kawai": nx.kamada_kawai_layout,
                "Planar": nx.planar_layout,
                "Random": nx.random_layout,
                "Spectral": nx.spectral_layout,
                "Spring": nx.spring_layout,
                "Shell": nx.shell_layout,
            }[layout](self.G_)

        max_width = max([dict_["width"] for _, _, dict_ in self.G_.edges.data()])
        for e in self.G_.edges.data():
            a, b, dict_ = e
            edge = [(a, b)]
            width = 0.2 + 4.0 * dict_["width"] / max_width
            nx.draw_networkx_edges(
                self.G_,
                pos=pos,
                ax=ax,
                edgelist=edge,
                width=width,
                edge_color="k",
                # with_labels=False,
                node_size=1,
                alpha=0.5,
            )

        for node in self.G_.nodes.data():
            nx.draw_networkx_nodes(
                self.G_,
                pos,
                ax=ax,
                nodelist=[node[0]],
                node_size=[node[1]["size"]],
                node_color=cluster_colors[node[1]["group"]],
                node_shape="o",
                edgecolors="k",
                linewidths=1,
                alpha=0.8,
            )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for label in self.top_terms_:
            x_point, y_point = pos[label]
            ax.text(
                x_point
                + 0.01 * (xlim[1] - xlim[0])
                + 0.001 * self.G_.nodes[label]["size"] / 300 * (xlim[1] - xlim[0]),
                y_point
                - 0.01 * (ylim[1] - ylim[0])
                - 0.001 * self.G_.nodes[label]["size"] / 300 * (ylim[1] - ylim[0]),
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

    def pyvis_plot(self):

        nt = Network_("700px", "870px", notebook=True)
        nt.from_nx(self.G_)

        for i, _ in enumerate(nt.nodes):
            if nt.nodes[i]["label"] not in self.top_terms_:
                nt.nodes[i]["label"] = ""

        for i, _ in enumerate(nt.nodes):
            nt.nodes[i]["size"] = nt.nodes[i]["size"] / 100

        return nt.show("net.html")
