"""
Plots
==================================================================================================

"""
import json
import textwrap
from os.path import dirname, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify
from techminer.chord_diagram import ChordDiagram
from wordcloud import ImageColorGenerator, WordCloud


TEXTLEN = 40

COLORMAPS = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "ocean",
    "gnuplot",
    "gnuplot2",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "twilight",
    "twilight_shifted",
    "hsv",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "flag",
    "prism",
    "gist_earth",
    "terrain",
    "gist_stern",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_ncar",
]


def bar(
    height,
    darkness=None,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    edgecolor="k",
    linewidth=0.5,
    zorder=10,
    **kwargs,
):
    """Make a bar plot.

    See https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.axes.Axes.bar.html.

    Args:
        height (pandas.Series): The height(s) of the bars.
        darkness (pandas.Series, optional): The color darkness of the bars. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "Greys".
        figsize (tuple, optional): Figure size passed to matplotlib. Defaults to (6, 6).
        fontsize (int, optional): Font size. Defaults to 11.
        edgecolor (str, optional): The colors of the bar edges. Defaults to "k".
        linewidth (float, optional): Width of the bar edges. If 0, don't draw edges. Defaults to 0.5.
        zorder (int, optional): order of drawing. Defaults to 10.

    Returns:
        container: Container with all the bars and optionally errorbars.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Num_Documents": [3, 2, 2, 1],
    ...         "Times_Cited": [1, 2, 3, 4],
    ...     },
    ...     index="author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> df
              Num_Documents  Times_Cited
    author 3              3            1
    author 1              2            2
    author 0              2            3
    author 2              1            4
    >>> fig = bar(height=df['Num_Documents'], darkness=df['Times_Cited'])
    >>> fig.savefig('/workspaces/techminer/sphinx/images/barplot1.png')

    .. image:: images/barplot1.png
        :width: 400px
        :align: center

    """
    darkness = height if darkness is None else darkness

    cmap = plt.cm.get_cmap(cmap)
    kwargs["color"] = [
        cmap(0.1 + 0.90 * (d - min(darkness)) / (max(darkness) - min(darkness)))
        for d in darkness
    ]

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.bar(
        x=range(len(height)),
        height=height,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    xticklabels = height.index
    if xticklabels.dtype != "int64":
        xticklabels = [
            textwrap.shorten(text=text, width=TEXTLEN) for text in xticklabels
        ]

    ax.set_xticks(np.arange(len(height)))
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", labelrotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.grid(axis="y", color="gray", linestyle=":")

    fig.set_tight_layout(True)

    return fig


def barh(
    width,
    darkness=None,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    edgecolor="k",
    linewidth=0.5,
    zorder=10,
    **kwargs,
):
    """Make a horizontal bar plot.

    See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.barh.html

    Args:
        width (pandas.Series): The widths of the bars.
        darkness (pandas.Series, optional): The color darkness of the bars. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "Greys".
        figsize (tuple, optional): Figure size passed to matplotlib. Defaults to (6, 6).
        fontsize (int, optional): Font size. Defaults to 11.
        edgecolor (str, optional): The colors of the bar edges. Defaults to "k".
        linewidth (float, optional): Width of the bar edges. If 0, don't draw edges. Defaults to 0.5.
        zorder (int, optional): order of drawing. Defaults to 10.

    Returns:
        container: Container with all the bars and optionally errorbars.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Num_Documents": [3, 2, 2, 1],
    ...         "Times_Cited": [1, 2, 3, 4],
    ...     },
    ...     index="author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> df
              Num_Documents  Times_Cited
    author 3              3            1
    author 1              2            2
    author 0              2            3
    author 2              1            4
    >>> fig = barh(width=df['Num_Documents'], darkness=df['Times_Cited'])
    >>> fig.savefig('/workspaces/techminer/sphinx/images/barhplot.png')

    .. image:: images/barhplot.png
        :width: 400px
        :align: center

    """
    darkness = width if darkness is None else darkness
    cmap = plt.cm.get_cmap(cmap)
    kwargs["color"] = [
        cmap(0.1 + 0.90 * (d - min(darkness)) / (max(darkness) - min(darkness)))
        for d in darkness
    ]

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.barh(
        y=range(len(width)),
        width=width,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    yticklabels = width.index
    if yticklabels.dtype != "int64":
        yticklabels = [
            textwrap.shorten(text=text, width=TEXTLEN) for text in yticklabels
        ]

    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(width)))
    ax.set_yticklabels(yticklabels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    ax.grid(axis="x", color="gray", linestyle=":")

    fig.set_tight_layout(True)

    return fig


def gant(
    X, cmap="Greys", figsize=(6, 6), fontsize=11, linewsidth=0.5, zorder=10, **kwargs,
):
    """

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "author 0": [1, 1, 0, 0, 0, 0, 0],
    ...         "author 1": [0, 1, 1, 0, 0, 0, 0],
    ...         "author 2": [1, 0, 0, 0, 0, 0, 0],
    ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
    ...         "author 4": [0, 0, 0, 0, 0, 0, 1],
    ...     },
    ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
    ... )
    >>> df
          author 0  author 1  author 2  author 3  author 4
    2010         1         0         1         0         0
    2011         1         1         0         0         0
    2012         0         1         0         1         0
    2013         0         0         0         1         0
    2014         0         0         0         1         0
    2015         0         0         0         0         0
    2016         0         0         0         0         1
    >>> fig = gant(df)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/gantplot.png')

    .. image:: images/gantplot.png
        :width: 400px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)

    data = X.copy()
    years = [year for year in range(data.index.min(), data.index.max() + 1)]
    data = data.applymap(lambda w: 1 if w > 0 else 0)
    data = data.applymap(lambda w: int(w))
    matrix1 = data.copy()
    matrix1 = matrix1.cumsum()
    matrix1 = matrix1.applymap(lambda X: True if X > 0 else False)
    matrix2 = data.copy()
    matrix2 = matrix2.sort_index(ascending=False)
    matrix2 = matrix2.cumsum()
    matrix2 = matrix2.applymap(lambda X: True if X > 0 else False)
    matrix2 = matrix2.sort_index(ascending=True)
    result = matrix1.eq(matrix2)
    result = result.applymap(lambda X: 1 if X is True else 0)
    gant_width = result.sum()
    #  gant_width = gant_width.map(lambda w: w - 0.5)
    gant_left = matrix1.applymap(lambda w: 1 - w)
    gant_left = gant_left.sum()
    gant_left = gant_left.map(lambda w: w - 0.5)

    w = pd.DataFrame({"terms": result.columns, "values": gant_width,})

    cmap = plt.cm.get_cmap(cmap)

    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    kwargs["color"] = [
        cmap(0.1 + 0.90 * (v - min(gant_width)) / (max(gant_width) - min(gant_width)))
        for v in gant_width
    ]

    ax.barh(
        y=range(len(data.columns)),
        width=gant_width,
        left=gant_left,
        edgecolor="k",
        linewidth=0.5,
        zorder=10,
        **kwargs,
    )

    xlim = ax.get_xlim()
    ax.set_xlim(left=xlim[0] - 0.5, right=xlim[0] + len(data) + 0.5)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.index)
    ax.tick_params(axis="x", labelrotation=90)

    ax.invert_yaxis()
    yticklabels = [textwrap.shorten(text=text, width=TEXTLEN) for text in data.columns]
    ax.set_yticks(np.arange(len(data.columns)))
    ax.set_yticklabels(yticklabels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.grid(axis="both", color="gray", linestyle=":")

    ax.set_aspect("equal")

    fig.set_tight_layout(True)

    return fig


def worldmap(
    x, cmap="Pastel2", figsize=(6, 6), legend=True, fontsize=11, *args, **kwargs,
):

    """Worldmap plot with the number of documents per country.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.Series(
    ...    data = [1000, 900, 800, 700, 600, 1000],
    ...    index = ["China", "Taiwan", "United States", "United Kingdom", "India", "Colombia"],
    ... )
    >>> x
    China             1000
    Taiwan             900
    United States      800
    United Kingdom     700
    India              600
    Colombia          1000
    dtype: int64

    >>> fig = worldmap(x, figsize=(15, 6))
    >>> fig.savefig('/workspaces/techminer/sphinx/images/worldmap.png')

    .. image:: images/worldmap.png
        :width: 2000px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    df = x.to_frame()

    df["color"] = x.map(lambda w: 0.1 + 0.9 * (w - x.min()) / (x.max() - x.min()))

    module_path = dirname(__file__)
    with open(join(module_path, "data/worldmap.data"), "r") as f:
        countries = json.load(f)
    for country in countries.keys():
        data = countries[country]
        for item in data:
            ax.plot(item[0], item[1], "-k", linewidth=0.5)
            if country in x.index.tolist():
                ax.fill(item[0], item[1], color=cmap(df.color[country]))
    #
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xleft = xmax - 0.02 * (xmax - xmin)
    xright = xmax
    xbar = np.linspace(xleft, xright, 10)
    ybar = np.linspace(ymin, ymin + (ymax - ymin), 100)
    xv, yv = np.meshgrid(xbar, ybar)
    z = yv / (ymax - ymin) - ymin
    ax.pcolormesh(xv, yv, z, cmap=cmap)
    #
    pos = np.linspace(ymin, ymin + (ymax - ymin), 11)
    value = [round(x.min() + (x.max() - x.min()) * i / 10, 0) for i in range(11)]
    for i in range(11):
        ax.text(
            xright + 0.4 * (xright - xleft),
            pos[i],
            str(int(value[i])),
            ha="left",
            va="center",
        )

    ax.plot(
        [xleft - 0.1 * (xright - xleft), xleft - 0.1 * (xright - xleft)],
        [ymin, ymax],
        color="gray",
        linewidth=1,
    )
    for i in range(11):
        ax.plot(
            [xleft - 0.0 * (xright - xleft), xright],
            [pos[i], pos[i]],
            linewidth=2.0,
            color=cmap((11 - i) / 11),
        )

    ax.set_aspect("equal")
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["bottom"].set_color("gray")
    ax.spines["top"].set_color("gray")
    ax.spines["right"].set_color("gray")
    ax.spines["left"].set_color("gray")

    fig.set_tight_layout(True)

    return fig


def pie(
    x,
    darkness=None,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    wedgeprops={
        "width": 0.6,
        "edgecolor": "k",
        "linewidth": 0.5,
        "linestyle": "-",
        "antialiased": True,
    },
    **kwargs,
):
    """Plot a pie chart.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Num_Documents": [3, 2, 2, 1],
    ...         "Times_Cited": [1, 2, 3, 4],
    ...     },
    ...     index="author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> df
              Num_Documents  Times_Cited
    author 3              3            1
    author 1              2            2
    author 0              2            3
    author 2              1            4
    >>> fig = pie(x=df['Num_Documents'], darkness=df['Times_Cited'], cmap="Blues")
    >>> fig.savefig('/workspaces/techminer/sphinx/images/pieplot.png')

    .. image:: images/pieplot.png
        :width: 400px
        :align: center


    """
    darkness = x if darkness is None else darkness

    cmap = plt.cm.get_cmap(cmap)
    colors = [
        cmap(0.1 + 0.90 * (d - min(darkness)) / (max(darkness) - min(darkness)))
        for d in darkness
    ]

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.pie(
        x=x, labels=x.index, colors=colors, wedgeprops=wedgeprops, **kwargs,
    )

    fig.set_tight_layout(True)

    return fig


def bubble(
    X,
    darkness=None,
    figsize=(6, 6),
    cmap="Greys",
    grid_lw=1.0,
    grid_c="gray",
    grid_ls=":",
    fontsize=11,
    **kwargs,
):

    """Creates a gant activity plot from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "author 0": [ 1, 2, 3, 4, 5, 6, 7],
    ...         "author 1": [14, 13, 12, 11, 10, 9, 8],
    ...         "author 2": [1, 5, 8, 9, 0, 0, 0],
    ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
    ...         "author 4": [0, 10, 0, 4, 2, 0, 1],
    ...     },
    ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
    ... )
    >>> df
          author 0  author 1  author 2  author 3  author 4
    2010         1        14         1         0         0
    2011         2        13         5         0        10
    2012         3        12         8         1         0
    2013         4        11         9         1         4
    2014         5        10         0         1         2
    2015         6         9         0         0         0
    2016         7         8         0         0         1

    >>> fig = bubble(df, axis=0, alpha=0.5, rmax=150)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/bubbleplot0.png')

    .. image:: images/bubbleplot0.png
        :width: 400px
        :align: center

    >>> fig = bubble(df, axis=1, alpha=0.5, rmax=150)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/bubbleplot1.png')

    .. image:: images/bubbleplot1.png
        :width: 400px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    x = X.copy()

    size_max = x.max().max()
    size_min = x.min().min()

    if darkness is None:
        darkness = x
    darkness = darkness.loc[:, x.columns]

    color_max = darkness.max().max()
    color_min = darkness.min().min()

    for idx, row in enumerate(x.index.tolist()):

        sizes = [
            150 + 1000 * (w - size_min) / (size_max - size_min) if w != 0 else 0
            for w in x.loc[row, :]
        ]

        colors = [
            cmap(0.2 + 0.8 * (w - color_min) / (color_max - color_min))
            for w in darkness.loc[row, :]
        ]

        #  return range(len(x.columns)), [idx] * len(x.columns)

        ax.scatter(
            list(range(len(x.columns))),
            [idx] * len(x.columns),
            marker="o",
            s=sizes,
            alpha=1.0,
            c=colors,
            edgecolors="k",
            zorder=11,
            #  **kwargs,
        )

    for idx, row in enumerate(x.iterrows()):
        ax.hlines(
            idx, -1, len(x.columns), linewidth=grid_lw, color=grid_c, linestyle=grid_ls,
        )

    for idx, col in enumerate(x.columns):
        ax.vlines(
            idx, -1, len(x.index), linewidth=grid_lw, color=grid_c, linestyle=grid_ls,
        )

    mean_color = 0.5 * (color_min + color_max)
    for idx_col, col in enumerate(x.columns):
        for idx_row, row in enumerate(x.index):

            if x[col][row] != 0:
                if darkness[col][row] >= 0.8 * mean_color:
                    text_color = "w"
                else:
                    text_color = "k"

                ax.text(
                    idx_col,
                    idx_row,
                    x[col][row],
                    va="center",
                    ha="center",
                    zorder=12,
                    color=text_color,
                )

    ax.set_aspect("equal")

    ax.set_xlim(-1, len(x.columns))
    ax.set_ylim(-1, len(x.index) + 1)

    ax.set_xticks(np.arange(len(x.columns)))
    ax.set_xticklabels(x.columns)
    ax.tick_params(axis="x", labelrotation=90)
    ax.xaxis.tick_top()

    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(x.index)))
    ax.set_yticklabels(x.index)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.set_tight_layout(True)

    return fig


def plot(
    data, cmap="Greys", figsize=(6, 6), fontsize=11, **kwargs,
):
    """Creates a plot from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "author 0": [ 1, 2, 3, 4, 5, 6, 7],
    ...         "author 1": [14, 13, 12, 11, 10, 9, 8],
    ...         "author 2": [1, 5, 8, 9, 0, 0, 0],
    ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
    ...         "author 4": [0, 10, 0, 4, 2, 0, 1],
    ...     },
    ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
    ... )
    >>> df
          author 0  author 1  author 2  author 3  author 4
    2010         1        14         1         0         0
    2011         2        13         5         0        10
    2012         3        12         8         1         0
    2013         4        11         9         1         4
    2014         5        10         0         1         2
    2015         6         9         0         0         0
    2016         7         8         0         0         1
    >>> fig = plot(df)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotplot.png')

    .. image:: images/plotplot.png
        :width: 400px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    x = data.copy()
    if "ID" in x.columns:
        x.pop("ID")
        ax.plot(
            range(len(x)),
            x[x.columns[1]],
            *args,
            scalex=scalex,
            scaley=scaley,
            **kwargs,
        )
        plt.xticks(
            np.arange(len(x[x.columns[0]])), x[x.columns[0]], rotation="vertical"
        )
        ax.set_xlabel(x.columns[0])
        ax.set_ylabel(x.columns[1])
    else:
        colors = [cmap(0.2 + i / (len(x.columns) - 1)) for i in range(len(x.columns))]

        for i, col in enumerate(x.columns):
            kwargs["color"] = colors[i]
            ax.plot(x.index, x[col], label=col, **kwargs)
        ax.legend()

    ax.set_xticks(x.index)
    ax.set_xticklabels(x.index)
    ax.tick_params(axis="x", labelrotation=90)
    # ax.xaxis.tick_top()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    fig.set_tight_layout(True)

    return fig


def wordcloud(
    x,
    darkness=None,
    figsize=(6, 6),
    font_path=None,
    width=400,
    height=200,
    margin=2,
    ranks_only=None,
    prefer_horizontal=0.9,
    mask=None,
    scale=1,
    max_words=200,
    min_font_size=4,
    stopwords=None,
    random_state=None,
    background_color="white",
    max_font_size=None,
    font_step=1,
    mode="RGB",
    relative_scaling="auto",
    regexp=None,
    collocations=True,
    cmap="Blues",
    normalize_plurals=True,
    contour_width=0,
    contour_color="black",
    repeat=False,
    include_numbers=False,
    min_word_length=0,
):
    """Plots a wordcloud from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Num_Documents": [10, 5, 2, 1],
    ...         "Times_Cited": [4, 3, 2, 0],
    ...     },
    ...     index = "author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> df
              Num_Documents  Times_Cited
    author 3             10            4
    author 1              5            3
    author 0              2            2
    author 2              1            0
    >>> fig = wordcloud(x=df['Num_Documents'], darkness=df['Times_Cited'])
    >>> fig.savefig('/workspaces/techminer/sphinx/images/wordcloud.png')

    .. image:: images/wordcloud.png
        :width: 400px
        :align: center

    """

    def color_func(word, font_size, position, orientation, font_path, random_state):
        return color_dic[word]

    darkness = x if darkness is None else darkness

    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    words = {key: value for key, value in zip(x.index, x)}

    color_dic = {
        key: cmap(
            0.5 + 0.5 * (value - darkness.min()) / (darkness.max() - darkness.min())
        )
        for key, value in zip(darkness.index, darkness)
    }

    for key in color_dic.keys():
        a, b, c, d = color_dic[key]
        color_dic[key] = (
            np.uint8(a * 255),
            np.uint8(b * 255),
            np.uint8(c * 255),
            np.uint8(d * 255),
        )

    wordcloud = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        margin=margin,
        ranks_only=ranks_only,
        prefer_horizontal=prefer_horizontal,
        mask=mask,
        scale=scale,
        color_func=color_func,
        max_words=max_words,
        min_font_size=min_font_size,
        stopwords=stopwords,
        random_state=random_state,
        background_color=background_color,
        max_font_size=max_font_size,
        font_step=font_step,
        mode=mode,
        relative_scaling=relative_scaling,
        regexp=regexp,
        collocations=collocations,
        colormap=cmap,
        normalize_plurals=normalize_plurals,
        contour_width=contour_width,
        contour_color=contour_color,
        repeat=repeat,
        include_numbers=include_numbers,
        min_word_length=min_word_length,
    )
    wordcloud.generate_from_frequencies(words)
    ax.imshow(wordcloud, interpolation="bilinear")
    #
    ax.spines["bottom"].set_color("lightgray")
    ax.spines["top"].set_color("lightgray")
    ax.spines["right"].set_color("lightgray")
    ax.spines["left"].set_color("lightgray")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_tight_layout(True)

    return fig


#
# def __gant(
#     x,
#     figsize=(8, 8),
#     fontsize=12,
#     grid_lw=1.0,
#     grid_c="gray",
#     grid_ls=":",
#     *args,
#     **kwargs,
# ):
#     """Creates a gant activity plot from a dataframe.
#
#     Examples
#     ----------------------------------------------------------------------------------------------
#
#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...         "author 0": [1, 1, 0, 0, 0, 0, 0],
#     ...         "author 1": [0, 1, 1, 0, 0, 0, 0],
#     ...         "author 2": [1, 0, 0, 0, 0, 0, 0],
#     ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
#     ...         "author 4": [0, 0, 0, 0, 0, 0, 1],
#     ...     },
#     ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
#     ... )
#     >>> df
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         0         1         0         0
#     2011         1         1         0         0         0
#     2012         0         1         0         1         0
#     2013         0         0         0         1         0
#     2014         0         0         0         1         0
#     2015         0         0         0         0         0
#     2016         0         0         0         0         1
#
#     >>> fig = gant(df)
#     >>> fig.savefig('/workspaces/techminer/sphinx/images/gantplot.png')
#
#     .. image:: images/gantplot.png
#         :width: 400px
#         :align: center
#
#     """
#     matplotlib.rc("font", size=fontsize)
#     fig = plt.Figure(figsize=figsize)
#     ax = fig.subplots()
#
#     x = x.copy()
#     if "linewidth" not in kwargs.keys() and "lw" not in kwargs.keys():
#         kwargs["linewidth"] = 4
#     if "marker" not in kwargs.keys():
#         kwargs["marker"] = "o"
#     if "markersize" not in kwargs.keys() and "ms" not in kwargs.keys():
#         kwargs["markersize"] = 8
#     if "color" not in kwargs.keys() and "c" not in kwargs.keys():
#         kwargs["color"] = "k"
#     for idx, col in enumerate(x.columns):
#         w = x[col]
#         w = w[w > 0]
#         ax.plot(w.index, [idx] * len(w.index), **kwargs)
#
#     ax.grid(axis="both", color=grid_c, linestyle=grid_ls, linewidth=grid_lw)
#
#     ax.set_yticks(np.arange(len(x.columns)))
#     ax.set_yticklabels(x.columns)
#     ax.invert_yaxis()
#
#     years = list(range(min(x.index), max(x.index) + 1))
#
#     ax.set_xticks(years)
#     ax.set_xticklabels(years)
#     ax.tick_params(axis="x", labelrotation=90)
#
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.spines["bottom"].set_visible(True)
#     ax.set_aspect("equal")
#
#     return fig
#


def heatmap(X, cmap="Greys", figsize=(6, 6), fontsize=11, **kwargs):
    """Plots a heatmap from a matrix.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         'word 0': [1.00, 0.80, 0.70, 0.00,-0.30],
    ...         'word 1': [0.80, 1.00, 0.70, 0.50, 0.00],
    ...         'word 2': [0.70, 0.70, 1.00, 0.00, 0.00],
    ...         'word 3': [0.00, 0.50, 0.00, 1.00, 0.30],
    ...         'word 4': [-0.30, 0.00, 0.00, 0.30, 1.00],
    ...     },
    ...     index=['word {:d}'.format(i) for i in range(5)]
    ... )
    >>> df
            word 0  word 1  word 2  word 3  word 4
    word 0     1.0     0.8     0.7     0.0    -0.3
    word 1     0.8     1.0     0.7     0.5     0.0
    word 2     0.7     0.7     1.0     0.0     0.0
    word 3     0.0     0.5     0.0     1.0     0.3
    word 4    -0.3     0.0     0.0     0.3     1.0
    >>> fig = heatmap(df)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap1.png')

    .. image:: images/plotheatmap1.png
        :width: 400px
        :align: center

    >>> fig = heatmap(df, cmap='Blues')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap2.png')

    .. image:: images/plotheatmap2.png
        :width: 400px
        :align: center


    >>> df = pd.DataFrame(
    ...     {
    ...         'word 0': [100, 80, 70, 0,30],
    ...         'word 1': [80, 100, 70, 50, 0],
    ...         'word 2': [70, 70, 100, 0, 0],
    ...         'word 3': [0, 50, 0, 100, 3],
    ...         'word 4': [30, 0, 0, 30, 100],
    ...     },
    ...     index=['word {:d}'.format(i) for i in range(5)]
    ... )
    >>> df
            word 0  word 1  word 2  word 3  word 4
    word 0     100      80      70       0      30
    word 1      80     100      70      50       0
    word 2      70      70     100       0       0
    word 3       0      50       0     100      30
    word 4      30       0       0       3     100
    >>> fig = heatmap(df, cmap='Greys')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap3.png')

    .. image:: images/plotheatmap3.png
        :width: 400px
        :align: center


    >>> df = pd.DataFrame(
    ...     {
    ...         'word 0': [100, 80, 70, 0,30, 1],
    ...         'word 1': [80, 100, 70, 50, 0, 2],
    ...         'word 2': [70, 70, 100, 0, 0, 3],
    ...         'word 3': [0, 50, 0, 100, 3, 4],
    ...         'word 4': [30, 0, 0, 30, 100, 5],
    ...     },
    ...     index=['word {:d}'.format(i) for i in range(6)]
    ... )
    >>> df
            word 0  word 1  word 2  word 3  word 4
    word 0     100      80      70       0      30
    word 1      80     100      70      50       0
    word 2      70      70     100       0       0
    word 3       0      50       0     100      30
    word 4      30       0       0       3     100
    word 5       1       2       3       4       5

    >>> fig = heatmap(df, cmap='Greys')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap3.png')

    .. image:: images/plotheatmap3.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    result = ax.pcolor(X.values, cmap=cmap, **kwargs,)
    X.columns = [
        textwrap.shorten(text=w, width=TEXTLEN) if isinstance(w, str) else w
        for w in X.columns
    ]
    X.index = [
        textwrap.shorten(text=w, width=TEXTLEN) if isinstance(w, str) else w
        for w in X.index
    ]

    ax.set_xticks(np.arange(len(X.columns)) + 0.5)
    ax.set_xticklabels(X.columns)
    ax.tick_params(axis="x", labelrotation=90)

    ax.set_yticks(np.arange(len(X.index)) + 0.5)
    ax.set_yticklabels(X.index)
    ax.invert_yaxis()

    cmap = plt.cm.get_cmap(cmap)

    if all(X.dtypes == "int64"):
        fmt = "{:3.0f}"
    else:
        fmt = "{:3.2f}"
    for idx_row, row in enumerate(X.columns):
        for idx_col, col in enumerate(X.index):
            if abs(X.loc[col, row]) > X.values.max().max() / 2.0:
                color = cmap(0.0)
            else:
                color = cmap(1.0)
            ax.text(
                idx_row + 0.5,
                idx_col + 0.5,
                fmt.format(X.loc[col, row]),
                ha="center",
                va="center",
                color=color,
            )
    ax.xaxis.tick_top()

    fig.set_tight_layout(True)

    return fig


def stacked_bar(
    X,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    edgecolor="k",
    linewidth=0.5,
    **kwargs,
):
    """Stacked vertical bar plot.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "col 0": [6, 5, 2, 3, 4, 1],
    ...         "col 1": [0, 1, 2, 3, 4, 5],
    ...         "col 2": [3, 2, 3, 1, 0, 1],
    ...     },
    ...     index = "author 0,author 1,author 2,author 3,author 4,author 5".split(","),
    ... )
    >>> df
              col 0  col 1  col 2
    author 0      6      0      3
    author 1      5      1      2
    author 2      2      2      3
    author 3      3      3      1
    author 4      4      4      0
    author 5      1      5      1

    >>> fig = stacked_bar(df, cmap='Blues')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/stkbar0.png')

    .. image:: images/stkbar0.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    bottom = X[X.columns[0]].map(lambda w: 0.0)

    for icol, col in enumerate(X.columns):

        kwargs["color"] = cmap((0.3 + 0.50 * icol / (len(X.columns) - 1)))
        ax.bar(
            x=range(len(X)), height=X[col], bottom=bottom, label=col, **kwargs,
        )
        bottom = bottom + X[col]

    ax.legend()

    ax.grid(axis="y", color="gray", linestyle=":")

    ax.set_xticks(np.arange(len(X)))
    ax.set_xticklabels(X.index)
    ax.tick_params(axis="x", labelrotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    fig.set_tight_layout(True)

    return fig


def stacked_barh(
    X, figsize=(6, 6), height=0.8, fontsize=11, cmap="Greys", **kwargs,
):
    """Stacked horzontal bar plot.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "col 0": [6, 5, 2, 3, 4, 1],
    ...         "col 1": [0, 1, 2, 3, 4, 5],
    ...         "col 2": [3, 2, 3, 1, 0, 1],
    ...     },
    ...     index = "author 0,author 1,author 2,author 3,author 4,author 5".split(","),
    ... )
    >>> df
              col 0  col 1  col 2
    author 0      6      0      3
    author 1      5      1      2
    author 2      2      2      3
    author 3      3      3      1
    author 4      4      4      0
    author 5      1      5      1
    >>> fig = stacked_barh(df, cmap='Blues')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/stkbarh1.png')

    .. image:: images/stkbarh1.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    left = X[X.columns[0]].map(lambda w: 0.0)

    for icol, col in enumerate(X.columns):

        kwargs["color"] = cmap((0.3 + 0.50 * icol / (len(X.columns) - 1)))
        ax.barh(
            y=range(len(X)),
            width=X[col],
            height=height,
            left=left,
            label=col,
            **kwargs,
        )
        left = left + X[col]

    ax.legend()

    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(X[X.columns[0]])))
    ax.set_yticklabels(X.index)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    ax.grid(axis="x", color="gray", linestyle=":")

    return fig


# def chord_diagram(
#     X,
#     node_sizes=None,
#     node_darkness=None,
#     figsize=(8, 8),
#     cmap="Greys",
#     alpha=1.0,
#     minval=0.0,
#     top_n_links=None,
#     solid_lines=False,
# ):
#     """Creates a chord diagram from a correlation or an auto-correlation matrix.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...         'word 0': [1.00, 0.80, 0.70, 0.00,-0.30],
#     ...         'word 1': [0.80, 1.00, 0.70, 0.50, 0.00],
#     ...         'word 2': [0.70, 0.70, 1.00, 0.00, 0.00],
#     ...         'word 3': [0.00, 0.50, 0.00, 1.00, 0.30],
#     ...         'word 4': [-0.30, 0.00, 0.00, 0.30, 1.00],
#     ...     },
#     ...     index=['word {:d}'.format(i) for i in range(5)]
#     ... )
#     >>> df
#             word 0  word 1  word 2  word 3  word 4
#     word 0     1.0     0.8     0.7     0.0    -0.3
#     word 1     0.8     1.0     0.7     0.5     0.0
#     word 2     0.7     0.7     1.0     0.0     0.0
#     word 3     0.0     0.5     0.0     1.0     0.3
#     word 4    -0.3     0.0     0.0     0.3     1.0
#     >>> fig = chord_diagram(df)
#     >>> fig.savefig('/workspaces/techminer/sphinx/images/plotcd1.png')

#     .. image:: images/plotcd1.png
#         :width: 400px
#         :align: center

#     >>> fig = chord_diagram(df, top_n_links=5)
#     >>> fig.savefig('/workspaces/techminer/sphinx/images/plotcd2.png')

#     .. image:: images/plotcd2.png
#         :width: 400px
#         :align: center

#     >>> fig = chord_diagram(df, solid_lines=True)
#     >>> fig.savefig('/workspaces/techminer/sphinx/images/plotcd3.png')

#     .. image:: images/plotcd3.png
#         :width: 400px
#         :align: center

#     """
#     # ---------------------------------------------------
#     #
#     # Node sizes
#     #
#     terms = X.columns

#     if node_sizes is None:
#         nod_sizes = [10] * len(X.columns)

#     if node_darkness is None:
#         node_darkness = [1] * len(X.columns)

#     max_size = max(node_sizes)
#     min_size = min(node_sizes)
#     if min_size == max_size:
#         node_sizes = [30] * len(terms)
#     else:
#         node_sizes = [
#             100 + int(1000 * (w - min_size) / (max_size - min_size)) for w in node_sizes
#         ]

#     #
#     # Node colors
#     #
#     cmap = plt.cm.get_cmap(cmap)
#     node_colors = [
#         cmap(0.2 + 0.75 * node_sizes[i] / max(node_sizes))
#         for i in range(len(node_sizes))
#     ]
#     #
#     # ---------------------------------------------------

#     cd = ChordDiagram()
#     for idx, term in enumerate(x.columns):
#         cd.add_node(term, color=node_colors[idx], s=node_sizes[idx])

#     if top_n_links is not None and top_n_links <= len(x.columns):
#         values = []
#         for idx_col in range(len(x.columns) - 1):
#             for idx_row in range(idx_col + 1, len(x.columns)):
#                 node_a = X.index[idx_row]
#                 node_b = X.columns[idx_col]
#                 value = X[node_b][node_a]
#                 values.append(value)
#         values = sorted(values, reverse=True)
#         minval = values[top_n_links - 1]

#     style = ["-", "-", "--", ":"]
#     if solid_lines is True:
#         style = list("----")

#     width = [4, 2, 1, 1]
#     if solid_lines is True:
#         width = [4, 2, 1, 1]

#     links = 0
#     for idx_col in range(len(X.columns) - 1):
#         for idx_row in range(idx_col + 1, len(X.columns)):

#             node_a = X.index[idx_row]
#             node_b = X.columns[idx_col]
#             value = X[node_b][node_a]

#             if value > 0.75 and value >= minval:
#                 cd.add_edge(
#                     node_a,
#                     node_b,
#                     linestyle=style[0],
#                     linewidth=width[0],
#                     color="black",
#                 )
#                 links += 1
#             elif value > 0.50 and value >= minval:
#                 cd.add_edge(
#                     node_a,
#                     node_b,
#                     linestyle=style[1],
#                     linewidth=width[1],
#                     color="black",
#                 )
#                 links += 1
#             elif value > 0.25 and value >= minval:
#                 cd.add_edge(
#                     node_a,
#                     node_b,
#                     linestyle=style[2],
#                     linewidth=width[2],
#                     color="black",
#                 )
#                 links += 1
#             elif value <= 0.25 and value >= minval and value > 0.0:
#                 cd.add_edge(
#                     node_a,
#                     node_b,
#                     linestyle=style[3],
#                     linewidth=width[3],
#                     color="black",
#                 )
#                 links += 1

#             if top_n_links is not None and links >= top_n_links:
#                 continue

#     return cd.plot(figsize=figsize)


def treemap(x, darkness=None, cmap="Greys", figsize=(6, 6), fontsize=11, alpha=0.9):
    """Creates a classification plot..

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.Series(
    ...     [10, 5, 2, 1],
    ...     index = "author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> x
    author 3    10
    author 1     5
    author 0     2
    author 2     1
    dtype: int64
    >>> fig = treemap(x)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/treeplot.png')

    .. image:: images/treeplot.png
        :width: 400px
        :align: center


    """
    darkness = x if darkness is None else darkness

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    labels = x.index
    labels = [textwrap.shorten(text=text, width=TEXTLEN) for text in labels]
    labels = [textwrap.wrap(text=text, width=15) for text in labels]
    labels = ["\n".join(text) for text in labels]

    colors = [
        cmap(0.4 + 0.6 * (d - darkness.min()) / (darkness.max() - darkness.min()))
        for d in darkness
    ]

    squarify.plot(
        sizes=x,
        label=labels,
        color=colors,
        alpha=alpha,
        ax=ax,
        pad=True,
        bar_kwargs={"edgecolor": "k", "linewidth": 0.5},
        text_kwargs={"color": "w", "fontsize": fontsize},
    )
    ax.axis("off")

    fig.set_tight_layout(True)

    return fig


#     def correspondence_plot(self):
#         """Computes and plot clusters of data using correspondence analysis.


#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     [[3, 8, 6, 1],
#         ...      [3, 4, 7, 5],
#         ...      [9, 9, 2, 3],
#         ...      [0, 1, 0, 4],
#         ...      [4, 4, 8, 7],
#         ...      [2, 4, 4, 9],
#         ...      [4, 2, 4, 1],
#         ...      [0, 0, 9, 2],
#         ...      [9, 3, 9, 4],
#         ...      [2, 3, 2, 0]],
#         ...     columns=list('ABCD'), index=list('abcdefghij'))
#         >>> df
#            A  B  C  D
#         a  3  8  6  1
#         b  3  4  7  5
#         c  9  9  2  3
#         d  0  1  0  4
#         e  4  4  8  7
#         f  2  4  4  9
#         g  4  2  4  1
#         h  0  0  9  2
#         i  9  3  9  4
#         j  2  3  2  0

#         >>> _ = Plot(df).correspondence_plot()
#         >>> plt.savefig('sphinx/images/corana0.png')

#         .. image:: images/corana0.png
#             :width: 500px
#             :align: center


#         """
#         n = self.df.values.sum()
#         P = self.df.values / n
#         column_masses = P.sum(axis=0)
#         row_masses = P.sum(axis=1)
#         E = np.outer(row_masses, column_masses)
#         R = P - E
#         I = R / E
#         Z = I * np.sqrt(E)
#         # u: left (rows),  s: singular values,  vh: right (columns)
#         u, d, v = np.linalg.svd(Z)
#         u = u[:, 0 : v.shape[0]]
#         std_coordinates_rows = u / np.sqrt(row_masses[:, None])
#         std_coordinates_cols = np.transpose(v) / np.sqrt(column_masses[None, :])
#         ppal_coordinates_rows = std_coordinates_rows * d[None, :]
#         ppal_coordinates_cols = std_coordinates_cols * d[:, None]
#         df_rows = pd.DataFrame(
#             ppal_coordinates_rows,
#             index=self.df.index,
#             columns=["f{:d}".format(i) for i in range(len(self.df.columns))],
#         )
#         df_columns = pd.DataFrame(
#             ppal_coordinates_cols,
#             index=self.df.columns,
#             columns=["f{:d}".format(i) for i in range(len(self.df.columns))],
#         )
#         result = pd.concat([df_columns, df_rows])

#         plt.clf()
#         plt.gca().scatter(
#             result.f0[: len(self.df.columns)], result.f1[: len(self.df.columns)]
#         )

#         plt.gca().scatter(
#             result.f0[len(self.df.columns) :],
#             result.f1[len(self.df.columns) :],
#             marker=".",
#         )

#         for i in range(len(self.df.columns)):
#             plt.text(result.f0[i], result.f1[i], self.df.columns[i])

#         plt.gca().spines["top"].set_visible(False)
#         plt.gca().spines["right"].set_visible(False)
#         plt.gca().spines["left"].set_visible(False)
#         plt.gca().spines["bottom"].set_visible(False)
#         return plt.gca()

#         # def encircle(x, y, ax=None, **kw):
#         #     p = np.c_[x, y]
#         #     hull = ConvexHull(p)
#         #     poly = plt.Polygon(p[hull.vertices, :], **kw)
#         #     plt.gca().add_patch(poly)
#         #     #

#         # plt.clf()
#         # plt.scatter(self.df[df.columns[0]], self.df[df.columns[1]], **kwargs)
#         # for i, index in enumerate(self.df.index):
#         #     plt.text(df[df.columns[0]][index], df[df.columns[1]][index], df.index[i])

#         # cluster = AgglomerativeClustering(
#         #     n_clusters=n_clusters, affinity="euclidean", linkage="ward"
#         # )
#         # df = df.copy()
#         # cluster.fit_predict(df)

#         # for icluster in range(n_clusters):
#         #     encircle(
#         #         df.loc[cluster.labels_ == icluster, df.columns[0]],
#         #         df.loc[cluster.labels_ == icluster, df.columns[1]],
#         #         alpha=0.2,
#         #         linewidth=0,
#         #     )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
