import numpy as np
import pandas as pd

from techminer.explode import __explode


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

    axis_to_sort = {0: [0], 1: [1], 2: [0, 1],}[axis]

    if sort_by == "Alphabetic":

        for m in axis_to_sort:
            X = X.sort_index(axis=m, ascending=ascending).sort_index(
                axis=m, ascending=ascending
            )

    elif sort_by == "Num Documents" or sort_by == "Times Cited":

        for m in axis_to_sort:
            X = sort_axis(
                data=X,
                num_documents=sort_by == "Num Documents",
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


def sort_axis(data, num_documents, axis, ascending):
    data = data.copy()
    if axis == 0:
        x = data.index.tolist()
    elif axis == 1:
        x = data.columns.tolist()
    else:
        raise NameError("Invalid axis value:" + str(axis))
    if num_documents is True:
        x = sorted(x, key=lambda w: w.split(" ")[-1], reverse=not ascending)
    else:
        x = sorted(
            x,
            key=lambda w: ":".join(w.split(" ")[-1].split(":")[::-1]),
            reverse=not ascending,
        )
    if isinstance(data, pd.DataFrame):
        if axis == 0:
            data = data.loc[x, :]
        else:
            data = data.loc[:, x]
    else:
        data = data[x]
    return data


def limit_to_exclude(data, axis, column, limit_to, exclude):

    data = data.copy()

    if axis == 0:
        new_axis = data.index
    elif axis == 1:
        new_axis = data.columns
    else:
        raise NameError("Invalid axis value:" + str(axis))

    #
    # Limit to
    #
    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        new_axis = [w for w in new_axis if w in limit_to]

    #
    # Exclude
    #
    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        new_axis = [w for w in new_axis if w not in exclude]

    if axis == 0:
        data = data.loc[new_axis, :]
    else:
        data = data.loc[:, new_axis]

    return data


def add_counters_to_axis(X, axis, data, column):

    X = X.copy()
    data = data.copy()
    data["Num_Documents"] = 1
    m = (
        __explode(data[[column, "Num_Documents", "Times_Cited", "ID"]], column)
        .groupby(column, as_index=True)
        .agg({"Num_Documents": np.sum, "Times_Cited": np.sum,})
    )
    n_Num_Documents = int(np.log10(m["Num_Documents"].max())) + 1
    n_Times_Cited = int(np.log10(m["Times_Cited"].max())) + 1
    fmt = "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"
    new_names = {
        key: fmt.format(key, int(nd), int(tc))
        for key, nd, tc in zip(m.index, m.Num_Documents, m.Times_Cited)
    }
    if axis == 0:
        X.index = [new_names[t] for t in X.index]
    elif axis == 1:
        X.columns = [new_names[t] for t in X.columns]
    else:
        raise NameError("Invalid axis value:" + str(axis))

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
        cmap(0.2 + 0.80 * (i - min_citations) / (max_citations - min_citations))
        for i in node_colors
    ]
    return node_colors
