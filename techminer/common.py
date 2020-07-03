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
