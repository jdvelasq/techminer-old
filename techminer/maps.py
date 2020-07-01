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

import pandas as pd

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as pyplot
from ipywidgets import AppLayout, GridspecLayout, Layout
import techminer.graph as graph
from sklearn.cluster import AgglomerativeClustering
from techminer.plots import COLORMAPS

pd.options.display.max_rows = 50
pd.options.display.max_columns = 50


def strategic_map(
    X, n_clusters=2, linkage="ward", output=0, cmap="Greys", figsize=(6, 6),
):

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(1 - X)
    cluster_dict = {key: value for key, value in zip(X.columns, clustering.labels_)}

    map = pd.DataFrame(
        {"cluster": list(range(n_clusters))}, index=list(range(n_clusters))
    )
    map["density"] = 0.0
    map["centrality"] = 0.0
    map["name"] = ""
    map["n_members"] = 0
    map["members"] = [[]] * len(map)

    #
    # Members of cluster
    #
    for t in X.columns:
        map.at[cluster_dict[t], "members"] = map.loc[cluster_dict[t], "members"] + [t]

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
    for i_cluster in range(n_clusters):
        Z = X[[t for t in X.columns if cluster_dict[t] == i_cluster]]
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
    for i_cluster in range(n_clusters):
        Z = X[[t for t in X.columns if cluster_dict[t] == i_cluster]]
        I = Z.copy()
        I = I.loc[[t for t in I.index if cluster_dict[t] != i_cluster], :]
        map.at[i_cluster, "centrality"] = I.sum().sum()

    #
    # Output
    #
    if output == 0:
        return map

    if output == 1:
        text = {}
        for i_cluster in range(n_clusters):
            text[map.name[i_cluster]] = map.members[i_cluster].split(";")
        return json.dumps(text, indent=4, sort_keys=True)

    if output == 2:

        matplotlib.rc("font", size=11)
        cmap = pyplot.cm.get_cmap(cmap)
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()

        node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in map.name]
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        node_sizes = [
            600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

        node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in map.name]
        max_citations = max(node_colors)
        min_citations = min(node_colors)
        node_colors = [
            cmap(0.2 + 0.80 * (i - min_citations) / (max_citations - min_citations))
            for i in node_colors
        ]

        ax.scatter(
            map.centrality,
            map.density,
            s=node_sizes,
            linewidths=1,
            edgecolors="k",
            c=node_colors,
        )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        dx = 0.15 * (xlim[1] - xlim[0])
        dy = 0.15 * (ylim[1] - ylim[0])

        ax.set_xlim(xlim[0] - dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] + dy)

        for idx, term in enumerate(map.name):
            x, y = map.centrality[idx], map.density[idx]
            ax.text(
                x
                + 0.01 * (xlim[1] - xlim[0])
                + 0.001 * node_sizes[idx] / 300 * (xlim[1] - xlim[0]),
                y
                - 0.01 * (ylim[1] - ylim[0])
                - 0.001 * node_sizes[idx] / 300 * (ylim[1] - ylim[0]),
                s=term,
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

        ax.axhline(
            y=map.density.median(),
            color="gray",
            linestyle="--",
            linewidth=0.5,
            zorder=-1,
        )
        ax.axvline(
            x=map.centrality.median(),
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

        #  ax.set_aspect("equal")
        ax.axis("off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        fig.set_tight_layout(True)

        return fig

    return None


MAP_COLUMNS = [
    "Author_Keywords",
    "Author_Keywords_CL",
    "Index_Keywords",
    "Index_Keywords_CL",
    "Abstract_words",
    "Title_words",
]


def __TAB0__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = [column for column in data.columns if column in MAP_COLUMNS]
    #
    left_panel = [
        # 0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Table", "Membership", "Plot"], layout=Layout(width="55%"),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="55%"),
            ),
        },
        # 2
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Times Cited",], layout=Layout(width="55%"),
            ),
        },
        # 3
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(20, 501, 5)), layout=Layout(width="55%"),
            ),
        },
        # 4
        {
            "arg": "normalization",
            "desc": "Normalization:",
            "widget": widgets.Dropdown(
                options=["None", "association", "inclusion", "jaccard", "salton"],
                layout=Layout(width="55%"),
            ),
        },
        # 5
        {
            "arg": "n_clusters",
            "desc": "# clusters:",
            "widget": widgets.Dropdown(
                options=list(range(2, 20)), layout=Layout(width="55%"),
            ),
        },
        # 6
        {
            "arg": "linkage",
            "desc": "Linkage:",
            "widget": widgets.Dropdown(
                options=["ward", "complete", "average", "single"],
                layout=Layout(width="55%"),
            ),
        },
        # 7
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
        },
        # 8
        {
            "arg": "width",
            "desc": "Fig Width",
            "widget": widgets.Dropdown(
                options=range(5, 20, 1), ensure_option=True, layout=Layout(width="55%"),
            ),
        },
        # 11
        {
            "arg": "height",
            "desc": "Fig Height",
            "widget": widgets.Dropdown(
                options=range(5, 20, 1), ensure_option=True, layout=Layout(width="55%"),
            ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        # Logic
        #
        view = kwargs["view"]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        normalization = kwargs["normalization"]
        n_clusters = int(kwargs["n_clusters"])
        linkage = kwargs["linkage"]
        cmap = kwargs["cmap"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        matrix = graph.co_occurrence_matrix(
            data=data,
            column=column,
            top_by=top_by,
            top_n=top_n,
            normalization=normalization,
            limit_to=limit_to,
            exclude=exclude,
        )

        strategic_map_table = strategic_map(
            X=matrix, n_clusters=n_clusters, linkage=linkage, figsize=(width, height),
        )

        output.clear_output()
        with output:
            if view == "Table":
                display(
                    strategic_map(
                        X=matrix,
                        n_clusters=n_clusters,
                        linkage=linkage,
                        output=0,
                        cmap=cmap,
                        figsize=(width, height),
                    )
                )

            if view == "Membership":
                display(
                    print(
                        strategic_map(
                            X=matrix,
                            n_clusters=n_clusters,
                            linkage=linkage,
                            cmap=cmap,
                            output=1,
                            figsize=(width, height),
                        )
                    )
                )

            if view == "Plot":
                display(
                    strategic_map(
                        X=matrix,
                        n_clusters=n_clusters,
                        linkage=linkage,
                        output=2,
                        cmap=cmap,
                        figsize=(width, height),
                    )
                )

        return

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    #
    grid = GridspecLayout(13, 6)
    #
    # Left panel
    #
    for index in range(len(left_panel)):
        grid[index, 0] = widgets.HBox(
            [
                widgets.Label(value=left_panel[index]["desc"]),
                left_panel[index]["widget"],
            ],
            layout=Layout(
                display="flex", justify_content="flex-end", align_content="center",
            ),
        )
    #
    # Output
    #
    grid[0:, 1:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )

    return grid


def app(data, limit_to=None, exclude=None, tab=None):
    """Jupyter Lab dashboard.
    """
    app_title = "Maps"
    tab_titles = [
        "Strategic Map",
    ]
    tab_list = [
        __TAB0__(data, limit_to=limit_to, exclude=exclude),
    ]

    if tab is not None:
        return AppLayout(
            header=widgets.HTML(
                value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                    app_title + " / " + tab_titles[tab]
                )
            ),
            center=tab_list[tab],
            pane_heights=["80px", "660px", 0],  # tamaño total de la ventana: Ok!
        )

    body = widgets.Tab()
    body.children = tab_list
    for i in range(len(tab_list)):
        body.set_title(i, tab_titles[i])
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                app_title
            )
        ),
        center=body,
        pane_heights=["80px", "720px", 0],
    )
