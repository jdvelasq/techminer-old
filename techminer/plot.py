"""
Plots
==================================================================================================




"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import ImageColorGenerator, WordCloud
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import squarify
from .chord_diagram import ChordDiagram


# ---------------------------------------------------------------------------------------------
# def cut_text(w):
#     if isinstance(w, (int, float)):
#         return w
#     return w if len(w) < 35 else w[:31] + '... ' + w[w.find('['):]
# #---------------------------------------------------------------------------------------------


class Plot:
    def __init__(self, df):
        self.df = df

    def heatmap(self, **kwargs):
        """Plots a heatmap from a matrix.

        Colormap availables are:
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'


        See more info in: 
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

        Examples
        ----------------------------------------------------------------------------------------------

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
        >>> _ = Plot(df).heatmap()
        >>> plt.savefig('guide/images/plotheatmap1.png')
        
        .. image:: images/plotheatmap1.png
            :width: 400px
            :align: center
        
        >>> _ = Plot(df).heatmap(cmap='Blues')
        >>> plt.savefig('guide/images/plotheatmap2.png')
        
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
        >>> _ = Plot(df).heatmap(cmap='Greys')
        >>> plt.savefig('guide/images/plotheatmap3.png')
        
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
        term 0     100      80      70       0      30
        term 1      80     100      70      50       0
        term 2      70      70     100       0       0
        term 3       0      50       0     100      30
        term 4      30       0       0       3     100
        term 5       1       2       3       4       5
        >>> _ = Plot(df).heatmap(cmap='Greys')
        >>> plt.savefig('guide/images/plotheatmap3.png')
        
        .. image:: images/plotheatmap3.png
            :width: 400px
            :align: center




        """
        plt.clf()
        x = self.df.copy()
        result = plt.gca().pcolor(np.transpose(x.values), **kwargs,)
        plt.xticks(np.arange(len(x.index)) + 0.5, x.index, rotation="vertical")
        plt.yticks(np.arange(len(x.columns)) + 0.5, x.columns)
        plt.gca().invert_yaxis()

        if "cmap" in kwargs:
            cmap = plt.cm.get_cmap(kwargs["cmap"])
        else:
            cmap = plt.cm.get_cmap()

        if x.values.max().max() <= 1.0 and x.values.min().min() >= -1:
            fmt = "{:3.2f}"
        else:
            fmt = "{:3.0f}"

        for idx_row, row in enumerate(x.index):
            for idx_col, col in enumerate(x.columns):

                if abs(x.at[row, col]) > x.values.max().max() / 2.0:
                    color = cmap(0.0)
                else:
                    color = cmap(1.0)

                plt.text(
                    idx_row + 0.5,
                    idx_col + 0.5,
                    fmt.format(x.at[row, col]),
                    ha="center",
                    va="center",
                    color=color,
                )
        plt.gca().xaxis.tick_top()

        return plt.gca()

    def chord_diagram(self, alpha=1.0, minval=0.0, top_n_links=None, solid_lines=False):
        """Creates a chord diagram from a correlation or an auto-correlation matrix.


        Examples
        ----------------------------------------------------------------------------------------------

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
        >>> _ = Plot(df).chord_diagram()
        >>> plt.savefig('guide/images/plotcd1.png')
        
        .. image:: images/plotcd1.png
            :width: 400px
            :align: center

        >>> _ = Plot(df).chord_diagram(top_n_links=5)
        >>> plt.savefig('guide/images/plotcd2.png')
        
        .. image:: images/plotcd2.png
            :width: 400px
            :align: center

        >>> _ = Plot(df).chord_diagram(solid_lines=True)
        >>> plt.savefig('guide/images/plotcd3.png')
        
        .. image:: images/plotcd3.png
            :width: 400px
            :align: center
        
        """
        plt.clf()

        x = self.df.copy()

        cd = ChordDiagram()
        cd.add_nodes_from(x.columns, color="black", s=40)

        if top_n_links is not None and top_n_links <= len(x.columns):
            values = []
            for idx_col in range(len(x.columns) - 1):
                for idx_row in range(idx_col + 1, len(x.columns)):
                    node_a = x.index[idx_row]
                    node_b = x.columns[idx_col]
                    value = x[node_b][node_a]
                    values.append(value)
            values = sorted(values, reverse=True)
            minval = values[top_n_links - 1]

        style = list("--::")
        if solid_lines is True:
            style = list("----")

        width = [2.5, 1, 1, 1]
        if solid_lines is True:
            width = [4, 2, 1, 1]

        links = 0
        for idx_col in range(len(x.columns) - 1):
            for idx_row in range(idx_col + 1, len(x.columns)):

                node_a = x.index[idx_row]
                node_b = x.columns[idx_col]
                value = x[node_b][node_a]

                if value > 0.75 and value >= minval:
                    cd.add_edge(
                        node_a,
                        node_b,
                        linestyle=style[0],
                        linewidth=width[0],
                        color="black",
                    )
                    links += 1
                elif value > 0.50 and value >= minval:
                    cd.add_edge(
                        node_a,
                        node_b,
                        linestyle=style[1],
                        linewidth=width[1],
                        color="black",
                    )
                    links += 1
                elif value > 0.25 and value >= minval:
                    cd.add_edge(
                        node_a,
                        node_b,
                        linestyle=style[2],
                        linewidth=width[2],
                        color="black",
                    )
                    links += 1
                elif value <= 0.25 and value >= minval:
                    cd.add_edge(
                        node_a,
                        node_b,
                        linestyle=style[3],
                        linewidth=width[3],
                        color="red",
                    )
                    links += 1

                if top_n_links is not None and links >= top_n_links:
                    continue

        return cd.plot()

    def tree(self, cmap="Blues", alpha=0.9):
        """Creates a classification plot from a dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
        ...         "Num Documents": [10, 5, 2, 1],
        ...         "ID": list(range(4)),
        ...     }
        ... )
        >>> df
            Authors  Num Documents  ID
        0  author 3             10   0
        1  author 1              5   1
        2  author 0              2   2
        3  author 2              1   3

        >>> _ = Plot(df).tree()
        >>> plt.savefig('guide/images/treeplot.png')
        
        .. image:: images/treeplot.png
            :width: 400px
            :align: center


        """
        plt.clf()
        x = self.df.copy()
        cmap = plt.cm.get_cmap(cmap)
        colors = [
            cmap((0.2 + 0.75 * x[x.columns[1]][i] / max(x[x.columns[1]])))
            for i in range(len(x[x.columns[1]]))
        ]
        squarify.plot(
            sizes=x[x.columns[1]], label=x[x.columns[0]], color=colors, alpha=alpha
        )
        plt.gca().axis("off")
        return plt.gca()

    def bubble(
        self,
        axis=0,
        rmax=80,
        cmap="Blues",
        grid_lw=1.0,
        grid_c="gray",
        grid_ls=":",
        **kwargs
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

        >>> _ = Plot(df).bubble(axis=0, alpha=0.5, rmax=150)
        >>> plt.savefig('guide/images/bubbleplot0.png')
        
        .. image:: images/bubbleplot0.png
            :width: 400px
            :align: center

        >>> _ = Plot(df).bubble(axis=1, alpha=0.5, rmax=150)
        >>> plt.savefig('guide/images/bubbleplot1.png')
        
        .. image:: images/bubbleplot1.png
            :width: 400px
            :align: center


        """
        plt.clf()
        cmap = plt.cm.get_cmap(cmap)
        x = self.df.copy()
        if axis == "index":
            axis == 0
        if axis == "columns":
            axis == 1

        vmax = x.max().max()
        vmin = x.min().min()

        rmin = 0

        if axis == 0:
            for idx, row in enumerate(x.iterrows()):
                values = [
                    10 * (rmin + (rmax - rmin) * w / (vmax - vmin))
                    for w in row[1].tolist()
                ]
                plt.gca().scatter(
                    range(len(x.columns)),
                    [idx] * len(x.columns),
                    marker="o",
                    s=values,
                    **kwargs,
                )
                plt.hlines(
                    idx,
                    -1,
                    len(x.columns),
                    linewidth=grid_lw,
                    color=grid_c,
                    linestyle=grid_ls,
                )
        else:
            for idx, col in enumerate(x.columns):
                values = [
                    10 * (rmin + (rmax - rmin) * w / (vmax - vmin)) for w in x[col]
                ]
                plt.gca().scatter(
                    [idx] * len(x.index),
                    range(len(x.index)),
                    marker="o",
                    s=values,
                    **kwargs,
                )
                plt.vlines(
                    idx,
                    -1,
                    len(x.index),
                    linewidth=grid_lw,
                    color=grid_c,
                    linestyle=grid_ls,
                )

        for idx_col, col in enumerate(x.columns):
            for idx_row, row in enumerate(x.index):

                if x[col][row] != 0:
                    plt.text(idx_col, idx_row, x[col][row], va="center", ha="center")

        plt.xlim(-1, len(x.columns))
        plt.ylim(-1, len(x.index) + 1)

        plt.xticks(
            np.arange(len(x.columns)),
            x.columns,
            rotation="vertical",
            horizontalalignment="center",
        )
        plt.yticks(np.arange(len(x.index)), x.index)

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)

        return plt.gca()

    def worldmap(self, cmap="Pastel2", legend=True, *args, **kwargs):
        """Worldmap plot with the number of documents per country.
        
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "Country": ["China", "Taiwan", "United States", "United Kingdom", "India", "Colombia"],
        ...         "Num Documents": [1000, 900, 800, 700, 600, 1000],
        ...     },
        ... )
        >>> df
                  Country  Num Documents
        0           China           1000
        1          Taiwan            900
        2   United States            800
        3  United Kingdom            700
        4           India            600
        5        Colombia           1000


        >>> _ = plt.figure(figsize=(10, 6))
        >>> _ = Plot(df).worldmap()
        >>> plt.savefig('guide/images/worldmap.png')
        
        .. image:: images/worldmap.png
            :width: 600px
            :align: center
        
        
        """
        plt.clf()
        cmap = plt.cm.get_cmap(cmap)
        x = self.df.copy()
        x[x.columns[0]] = x[x.columns[0]].map(
            lambda w: w.replace("United States", "United States of America")
        )
        world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        world = world[world.name != "Antarctica"]
        world["q"] = 0
        world.index = world.name
        for _, row in x.iterrows():
            if row[0] in world.index:
                world.at[row[0], "q"] = row[1]
        axx = plt.gca()
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        world.plot(column="q", cmap=cmap, legend=legend, ax=axx, cax=cax, **kwargs)
        return plt.gca()

    def gant(self, hlines_lw=0.5, hlines_c="gray", hlines_ls=":", *args, **kwargs):

        """Creates a gant activity plot from a dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> pd = pd.DataFrame(
        ...     {
        ...         "author 0": [1, 1, 0, 0, 0, 0, 0],
        ...         "author 1": [0, 1, 1, 0, 0, 0, 0],
        ...         "author 2": [1, 0, 0, 0, 0, 0, 0],
        ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
        ...         "author 4": [0, 0, 0, 0, 0, 0, 1],
        ...     },
        ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
        ... )
        >>> pd
              author 0  author 1  author 2  author 3  author 4
        2010         1         0         1         0         0
        2011         1         1         0         0         0
        2012         0         1         0         1         0
        2013         0         0         0         1         0
        2014         0         0         0         1         0
        2015         0         0         0         0         0
        2016         0         0         0         0         1

        >>> _ = Plot(pd).gant()
        >>> plt.savefig('guide/images/gantplot.png')
        
        .. image:: images/gantplot.png
            :width: 400px
            :align: center


        """
        plt.clf()
        x = self.df.copy()
        if "linewidth" not in kwargs.keys() and "lw" not in kwargs.keys():
            kwargs["linewidth"] = 4
        if "marker" not in kwargs.keys():
            kwargs["marker"] = "o"
        if "markersize" not in kwargs.keys() and "ms" not in kwargs.keys():
            kwargs["markersize"] = 10
        if "color" not in kwargs.keys() and "c" not in kwargs.keys():
            kwargs["color"] = "k"
        for idx, col in enumerate(x.columns):
            w = x[col]
            w = w[w > 0]
            plt.gca().plot(w.index, [idx] * len(w.index), **kwargs)
        plt.hlines(
            range(len(x.columns)),
            x.index.min(),
            x.index.max(),
            linewidth=hlines_lw,
            color=hlines_c,
            linestyle=hlines_ls,
        )
        plt.yticks(np.arange(len(x.columns)), x.columns)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        return plt.gca()

    def pie(
        self,
        cmap="Greys",
        explode=None,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=1.1,
        startangle=None,
        radius=None,
        counterclock=True,
        wedgeprops=None,
        textprops=None,
        center=(0, 0),
        frame=False,
        rotatelabels=False,
    ):
        """Creates a pie plot from a dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> pd = pd.DataFrame(
        ...     {
        ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
        ...         "Num Documents": [3, 2, 2, 1],
        ...         "ID": list(range(4)),
        ...     }
        ... )
        >>> pd
            Authors  Num Documents  ID
        0  author 3              3   0
        1  author 1              2   1
        2  author 0              2   2
        3  author 2              1   3
        >>> _ = Plot(pd).pie(cmap=plt.cm.Blues)
        >>> plt.savefig('guide/images/pieplot.png')
        
        .. image:: images/pieplot.png
            :width: 400px
            :align: center


        """
        plt.clf()
        cmap = plt.cm.get_cmap(cmap)
        x = self.df.copy()
        x.pop("ID")
        colors = None
        if cmap is not None:
            colors = [
                cmap(1.0 - 0.9 * (i / len(x))) for i in range(len(x[x.columns[1]]))
            ]
        plt.gca().pie(
            x=x[x.columns[1]],
            explode=explode,
            labels=x[x.columns[0]],
            colors=colors,
            autopct=autopct,
            pctdistance=pctdistance,
            shadow=shadow,
            labeldistance=labeldistance,
            startangle=startangle,
            radius=radius,
            counterclock=counterclock,
            wedgeprops=wedgeprops,
            textprops=textprops,
            center=center,
            frame=frame,
            rotatelabels=rotatelabels,
        )
        return plt.gca()

    def bar(self, width=0.8, bottom=None, align="center", cmap="Greys", **kwargs):
        """Creates a bar plot from a dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> pd = pd.DataFrame(
        ...     {
        ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
        ...         "Num Documents": [3, 2, 2, 1],
        ...         "ID": list(range(4)),
        ...     }
        ... )
        >>> pd
            Authors  Num Documents  ID
        0  author 3              3   0
        1  author 1              2   1
        2  author 0              2   2
        3  author 2              1   3
        >>> _ = Plot(pd).bar(cmap=plt.cm.Blues)
        >>> plt.savefig('guide/images/barplot.png')
        
        .. image:: images/barplot.png
            :width: 400px
            :align: center


        """
        plt.clf()
        cmap = plt.cm.get_cmap(cmap)
        x = self.df.copy()
        x.pop("ID")
        if cmap is not None:
            kwargs["color"] = [
                cmap((0.2 + 0.75 * x[x.columns[1]][i] / max(x[x.columns[1]])))
                for i in range(len(x[x.columns[1]]))
            ]
        result = plt.gca().bar(
            x=range(len(x)),
            height=x[x.columns[1]],
            width=width,
            bottom=bottom,
            align=align,
            **({}),
            **kwargs,
        )
        plt.xticks(
            np.arange(len(x[x.columns[0]])), x[x.columns[0]], rotation="vertical"
        )
        plt.xlabel(x.columns[0])
        plt.ylabel(x.columns[1])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        return plt.gca()

    def barh(self, height=0.8, left=None, align="center", cmap=None, **kwargs):
        """Make a pie chart from a dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> pd = pd.DataFrame(
        ...     {
        ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
        ...         "Num Documents": [3, 2, 2, 1],
        ...         "ID": list(range(4)),
        ...     }
        ... )
        >>> pd
            Authors  Num Documents  ID
        0  author 3              3   0
        1  author 1              2   1
        2  author 0              2   2
        3  author 2              1   3
        >>> _ = Plot(pd).barh(cmap=plt.cm.Blues)
        >>> plt.savefig('guide/images/barhplot.png')
        
        .. image:: images/barhplot.png
            :width: 400px
            :align: center
        
        """
        plt.clf()
        x = self.df.copy()
        x.pop("ID")
        if cmap is not None:
            cmap = plt.cm.get_cmap(cmap)
            kwargs["color"] = [
                cmap((0.2 + 0.75 * x[x.columns[1]][i] / max(x[x.columns[1]])))
                for i in range(len(x[x.columns[1]]))
            ]
        plt.gca().barh(
            y=range(len(x)),
            width=x[x.columns[1]],
            height=height,
            left=left,
            align=align,
            **kwargs,
        )
        plt.gca().invert_yaxis()
        plt.yticks(np.arange(len(x[x.columns[0]])), x[x.columns[0]])
        plt.xlabel(x.columns[1])
        plt.ylabel(x.columns[0])
        #
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        #
        return plt.gca()

    def plot(self, *args, scalex=True, scaley=True, **kwargs):
        """Creates a plot from a dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> pd = pd.DataFrame(
        ...     {
        ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
        ...         "Num Documents": [3, 2, 2, 1],
        ...         "ID": list(range(4)),
        ...     }
        ... )
        >>> pd
            Authors  Num Documents  ID
        0  author 3              3   0
        1  author 1              2   1
        2  author 0              2   2
        3  author 2              1   3
        >>> _ = Plot(pd).plot()
        >>> plt.savefig('guide/images/plotplot.png')
        
        .. image:: images/plotplot.png
            :width: 400px
            :align: center


        """
        plt.clf()
        x = self.df.copy()
        x.pop("ID")
        plt.gca().plot(
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
        plt.xlabel(x.columns[0])
        plt.ylabel(x.columns[1])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        return plt.gca()

    def wordcloud(
        self,
        font_path=None,
        width=400,
        height=200,
        margin=2,
        ranks_only=None,
        prefer_horizontal=0.9,
        mask=None,
        scale=1,
        color_func=None,
        max_words=200,
        min_font_size=4,
        stopwords=None,
        random_state=None,
        background_color="black",
        max_font_size=None,
        font_step=1,
        mode="RGB",
        relative_scaling="auto",
        regexp=None,
        collocations=True,
        colormap=None,
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
        >>> pd = pd.DataFrame(
        ...     {
        ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
        ...         "Num Documents": [10, 5, 2, 1],
        ...         "ID": list(range(4)),
        ...     }
        ... )
        >>> pd
            Authors  Num Documents  ID
        0  author 3             10   0
        1  author 1              5   1
        2  author 0              2   2
        3  author 2              1   3
        >>> _ = Plot(pd).wordcloud()
        >>> plt.savefig('guide/images/wordcloud.png')        
        
        .. image:: images/wordcloud.png
            :width: 400px
            :align: center     
        """
        plt.clf()
        x = self.df.copy()
        x.pop("ID")
        words = {row[0]: row[1] for _, row in x.iterrows()}
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
            colormap=colormap,
            normalize_plurals=normalize_plurals,
            contour_width=contour_width,
            contour_color=contour_color,
            repeat=repeat,
            include_numbers=include_numbers,
            min_word_length=min_word_length,
        )
        wordcloud.generate_from_frequencies(words)
        plt.gca().imshow(wordcloud, interpolation="bilinear")
        plt.gca().axis("off")
        return plt.gca()
