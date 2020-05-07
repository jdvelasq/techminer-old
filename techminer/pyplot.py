"""
TechMiner.Pyplot
==================================================================================================




"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import ImageColorGenerator, WordCloud
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Pyplot:
    def __init__(self, df):
        self.df = df

    def bubble(
        self,
        axis=0,
        rmax=80,
        cmap=plt.cm.Blues,
        grid_lw=1.0,
        grid_c="gray",
        grid_ls=":",
        **kwargs
    ):

        """Creates a gant activity plot from a dataframe.

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

        >>> _ = Pyplot(df).bubble(axis=0, alpha=0.5, rmax=150)
        >>> plt.savefig('guide/images/bubbleplot0.png')
        
        .. image:: images/bubbleplot0.png
            :width: 400px
            :align: center

        >>> _ = Pyplot(df).bubble(axis=1, alpha=0.5, rmax=150)
        >>> plt.savefig('guide/images/bubbleplot1.png')
        
        .. image:: images/bubbleplot1.png
            :width: 400px
            :align: center


        """
        x = self.df.copy()
        if axis == "index":
            axis == 0
        if axis == "columns":
            axis == 1
        plt.clf()

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

    def worldmap(self, cmap=plt.cm.Pastel2, legend=True, *args, **kwargs):
        """Worldmap plot with the number of documents per country.
        
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
        >>> _ = Pyplot(df).worldmap()
        >>> plt.savefig('guide/images/worldmap.png')
        
        .. image:: images/worldmap.png
            :width: 600px
            :align: center
        
        
        """
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

        >>> _ = Pyplot(pd).gant()
        >>> plt.savefig('guide/images/gantplot.png')
        
        .. image:: images/gantplot.png
            :width: 400px
            :align: center


        """
        x = self.df.copy()
        plt.clf()
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
        cmap=plt.cm.Greys,
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
        >>> _ = Pyplot(pd).pie(cmap=plt.cm.Blues)
        >>> plt.savefig('guide/images/pieplot.png')
        
        .. image:: images/pieplot.png
            :width: 400px
            :align: center


        """
        x = self.df.copy()
        x.pop("ID")
        colors = None
        if cmap is not None:
            colors = [
                cmap(1.0 - 0.9 * (i / len(x))) for i in range(len(x[x.columns[1]]))
            ]
        plt.clf()
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

    def bar(self, width=0.8, bottom=None, align="center", cmap=plt.cm.Greys, **kwargs):
        """Creates a bar plot from a dataframe.

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
        >>> _ = Pyplot(pd).bar(cmap=plt.cm.Blues)
        >>> plt.savefig('guide/images/barplot.png')
        
        .. image:: images/barplot.png
            :width: 400px
            :align: center


        """
        x = self.df.copy()
        x.pop("ID")
        if cmap is not None:
            kwargs["color"] = [
                cmap((0.2 + 0.75 * x[x.columns[1]][i] / max(x[x.columns[1]])))
                for i in range(len(x[x.columns[1]]))
            ]
        plt.clf()
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
        >>> _ = Pyplot(pd).barh(cmap=plt.cm.Blues)
        >>> plt.savefig('guide/images/barhplot.png')
        
        .. image:: images/barhplot.png
            :width: 400px
            :align: center
        
        """
        x = self.df.copy()
        x.pop("ID")
        if cmap is not None:
            kwargs["color"] = [
                cmap((0.2 + 0.75 * x[x.columns[1]][i] / max(x[x.columns[1]])))
                for i in range(len(x[x.columns[1]]))
            ]
        plt.clf()
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
        >>> _ = Pyplot(pd).plot()
        >>> plt.savefig('guide/images/plotplot.png')
        
        .. image:: images/plotplot.png
            :width: 400px
            :align: center


        """
        x = self.df.copy()
        x.pop("ID")
        plt.clf()
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
        >>> _ = Pyplot(pd).wordcloud()
        >>> plt.savefig('guide/images/wordcloud.png')        
        
        .. image:: images/wordcloud.png
            :width: 400px
            :align: center     
        """
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
        plt.clf()
        plt.gca().imshow(wordcloud, interpolation="bilinear")
        plt.gca().axis("off")
        return plt.gca()


#     # ----------------------------------------------------------------------------------------------------
#     def heatmap(
#         self,
#         ascending_r=None,
#         ascending_c=None,
#         alpha=None,
#         norm=None,
#         cmap=None,
#         vmin=None,
#         vmax=None,
#         data=None,
#         **kwargs
#     ):
#         """Heat map.


#         https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'


#         >>> from techminer.datasets import load_test_cleaned
#         >>> from techminer.dataframe import DataFrame
#         >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
#         >>> result = rdf.co_ocurrence(column_r='Authors', column_c='Document Type', top_n=5)
#         >>> from techminer.plot import Plot
#         >>> Plot(result).heatmap()

#         .. image:: ../figs//heatmap.jpg
#             :width: 600px
#             :align: center

#         """

#         x = self.pdf.copy()
#         x.pop("ID")
#         x = pd.pivot_table(
#             data=x,
#             index=x.columns[0],
#             columns=x.columns[1],
#             margins=False,
#             fill_value=0,
#         )
#         x.columns = [b for _, b in x.columns]
#         result = plt.gca().pcolor(
#             x.values,
#             alpha=alpha,
#             norm=norm,
#             cmap=cmap,
#             vmin=vmin,
#             vmax=vmax,
#             data=data,
#             **({}),
#             **kwargs,
#         )
#         plt.xticks(np.arange(len(x.index)) + 0.5, x.index, rotation="vertical")
#         plt.yticks(np.arange(len(x.columns)) + 0.5, x.columns)
#         plt.gca().invert_yaxis()

#         return

#         # ## force the same order of cells in rows and cols ------------------------------------------
#         # if self._call == 'auto_corr':
#         #     if ascending_r is None and ascending_c is None:
#         #         ascending_r = True
#         #         ascending_c = True
#         #     elif ascending_r is not None and ascending_r != ascending_c:
#         #         ascending_c = ascending_r
#         #     elif ascending_c is not None and ascending_c != ascending_r:
#         #         ascending_r = ascending_c
#         #     else:
#         #         pass
#         # ## end -------------------------------------------------------------------------------------

#         x = self.tomatrix(ascending_r, ascending_c)

#         ## rename columns and row index
#         # x.columns = [cut_text(w) for w in x.columns]
#         # x.index = [cut_text(w) for w in x.index]

#         # if self._call == 'factor_analysis':
#         #     x = self.tomatrix(ascending_r, ascending_c)
#         #     x = x.transpose()
#         #     plt.pcolor(np.transpose(abs(x.values)), cmap=cmap)
#         # else:
#         #     plt.pcolor(np.transpose(x.values), cmap=cmap)

#         # plt.xticks(np.arange(len(x.index))+0.5, x.index, rotation='vertical')
#         # plt.yticks(np.arange(len(x.columns))+0.5, x.columns)
#         # plt.gca().invert_yaxis()

#         ## changes the color of rectangle for autocorrelation heatmaps ---------------------------

#         # if self._call == 'auto_corr':
#         #     for idx in np.arange(len(x.index)):
#         #         plt.gca().add_patch(
#         #             Rectangle((idx, idx), 1, 1, fill=False, edgecolor='red')
#         #         )

#         ## end ------------------------------------------------------------------------------------

#         ## annotation
#         # for idx_row, row in enumerate(x.index):
#         #     for idx_col, col in enumerate(x.columns):

#         #         if self._call in ['auto_corr', 'cross_corr', 'factor_analysis']:

#         #             if abs(x.at[row, col]) > x.values.max() / 2.0:
#         #                 color = 'white'
#         #             else:
#         #                 color = 'black'

#         #             plt.text(
#         #                 idx_row + 0.5,
#         #                 idx_col + 0.5,
#         #                 "{:3.2f}".format(x.at[row, col]),
#         #                 ha="center",
#         #                 va="center",
#         #                 color=color)

#         #         else:
#         #             if x.at[row, col] > 0:

#         #                 if x.at[row, col] > x.values.max() / 2.0:
#         #                     color = 'white'
#         #                 else:
#         #                     color = 'black'

#         #                 plt.text(
#         #                     idx_row + 0.5,
#         #                     idx_col + 0.5,
#         #                     int(x.at[row, col]),
#         #                     ha="center",
#         #                     va="center",
#         #                     color=color)

#         # plt.tight_layout()
#         # plt.show()
