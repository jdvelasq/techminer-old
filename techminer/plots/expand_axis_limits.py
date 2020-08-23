def expand_axis_limits(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = 0.15 * (xlim[1] - xlim[0])
    dy = 0.15 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - dx, xlim[1] + dx)
    ax.set_ylim(ylim[0] - dy, ylim[1] + dy)
