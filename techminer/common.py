import numpy as np
import pandas as pd

from techminer.explode import explode


#
# refactor common conde
#
def ax_expand_limits(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = 0.15 * (xlim[1] - xlim[0])
    dy = 0.15 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - dx, xlim[1] + dx)
    ax.set_ylim(ylim[0] - dy, ylim[1] + dy)


def ax_text_node_labels(ax, labels, dict_pos, node_sizes):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for idx, label in enumerate(labels):
        x_point, y_point = dict_pos[label]
        ax.text(
            x_point
            + 0.01 * (xlim[1] - xlim[0])
            + 0.001 * node_sizes[idx] / 300 * (xlim[1] - xlim[0]),
            y_point
            - 0.01 * (ylim[1] - ylim[0])
            - 0.001 * node_sizes[idx] / 300 * (ylim[1] - ylim[0]),
            s=label,
            fontsize=10,
            bbox=dict(
                facecolor="w", alpha=1.0, edgecolor="gray", boxstyle="round,pad=0.5",
            ),
            horizontalalignment="left",
            verticalalignment="top",
        )


def set_ax_splines_invisible(ax, exclude=None):
    for x in ["top", "right", "left", "bottom"]:
        if exclude is None or x != exclude:
            ax.spines[x].set_visible(False)


def sort_by_axis(data, sort_by, ascending, axis):

    X = data.copy()
    #  sort_by = sort_by.replace(' ', '_').replace('-','_').replace('/','_').replace('(', '').replace(')', '')

    axis_to_sort = {0: [0], 1: [1], 2: [0, 1],}[axis]

    if sort_by == "Alphabetic":

        for m in axis_to_sort:
            X = X.sort_index(axis=m, ascending=ascending).sort_index(
                axis=m, ascending=ascending
            )

    elif (
        sort_by == "Num Documents"
        or sort_by == "Times Cited"
        or sort_by == "Num_Documents"
        or sort_by == "Times_Cited"
    ):

        for m in axis_to_sort:
            X = sort_axis(
                data=X,
                num_documents=(sort_by == "Num_Documents")
                or (sort_by == "Num Documents"),
                axis=m,
                ascending=ascending,
            )

    elif sort_by == "Data":

        for m in axis_to_sort:
            if m == 0:
                t = X.max(axis=1)
                X = X.loc[t.sort_values(ascending=ascending).index, :]
            else:
                t = X.max(axis=0)
                X = X.loc[:, t.sort_values(ascending=ascending).index]
    else:

        raise NameError("Invalid 'Sort by' value:" + sort_by)

    return X


def counters_to_node_sizes(x):
    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in x]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    node_sizes = [
        500 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
    ]
    return node_sizes


def counters_to_node_colors(x, cmap):
    node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in x]
    max_citations = max(node_colors)
    min_citations = min(node_colors)
    node_colors = [
        cmap(0.4 + 0.60 * (i - min_citations) / (max_citations - min_citations))
        for i in node_colors
    ]
    return node_colors
