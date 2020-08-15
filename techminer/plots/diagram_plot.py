import matplotlib
import matplotlib.pyplot as pyplot

import techminer.common as cmn


def diagram_plot(x, y, labels, x_axis_at=0, y_axis_at=0, cmap=None, width=5, height=5):

    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=(width, height))
    ax = fig.subplots()
    cmap = pyplot.cm.get_cmap(cmap)

    node_sizes = cmn.counters_to_node_sizes(labels)
    node_colors = cmn.counters_to_node_colors(x=labels, cmap=cmap)

    ax.scatter(
        x, y, s=node_sizes, linewidths=1, edgecolors="k", c=node_colors,
    )

    cmn.ax_expand_limits(ax)

    cmn.ax_text_node_labels(
        ax,
        labels=labels,
        dict_pos={key: (c, d) for key, c, d in zip(labels, x, y,)},
        node_sizes=node_sizes,
    )

    ax.axhline(
        y=y_axis_at, color="gray", linestyle="--", linewidth=0.5, zorder=-1,
    )
    ax.axvline(
        x=x_axis_at, color="gray", linestyle="--", linewidth=1, zorder=-1,
    )

    ax.axis("off")

    cmn.set_ax_splines_invisible(ax)

    fig.set_tight_layout(True)

    return fig
