"""
TechMiner.Plot
==================================================================================================




"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, pdf):
        self.pdf = pdf

    # ----------------------------------------------------------------------------------------------------
    def heatmap(
        self,
        ascending_r=None,
        ascending_c=None,
        alpha=None,
        norm=None,
        cmap=None,
        vmin=None,
        vmax=None,
        data=None,
        **kwargs
    ):
        """Heat map.


        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'


        >>> from techminer.datasets import load_test_cleaned
        >>> from techminer.dataframe import DataFrame
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> result = rdf.co_ocurrence(column_r='Authors', column_c='Document Type', top_n=5)
        >>> from techminer.plot import Plot
        >>> Plot(result).heatmap()

        .. image:: ../figs//heatmap.jpg
            :width: 600px
            :align: center

        """

        x = self.pdf.copy()
        x.pop("ID")
        x = pd.pivot_table(
            data=x,
            index=x.columns[0],
            columns=x.columns[1],
            margins=False,
            fill_value=0,
        )
        x.columns = [b for _, b in x.columns]
        result = plt.gca().pcolor(
            x.values,
            alpha=alpha,
            norm=norm,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            data=data,
            **({}),
            **kwargs,
        )
        plt.xticks(np.arange(len(x.index)) + 0.5, x.index, rotation="vertical")
        plt.yticks(np.arange(len(x.columns)) + 0.5, x.columns)
        plt.gca().invert_yaxis()

        return

        # ## force the same order of cells in rows and cols ------------------------------------------
        # if self._call == 'auto_corr':
        #     if ascending_r is None and ascending_c is None:
        #         ascending_r = True
        #         ascending_c = True
        #     elif ascending_r is not None and ascending_r != ascending_c:
        #         ascending_c = ascending_r
        #     elif ascending_c is not None and ascending_c != ascending_r:
        #         ascending_r = ascending_c
        #     else:
        #         pass
        # ## end -------------------------------------------------------------------------------------

        x = self.tomatrix(ascending_r, ascending_c)

        ## rename columns and row index
        # x.columns = [cut_text(w) for w in x.columns]
        # x.index = [cut_text(w) for w in x.index]

        # if self._call == 'factor_analysis':
        #     x = self.tomatrix(ascending_r, ascending_c)
        #     x = x.transpose()
        #     plt.pcolor(np.transpose(abs(x.values)), cmap=cmap)
        # else:
        #     plt.pcolor(np.transpose(x.values), cmap=cmap)

        # plt.xticks(np.arange(len(x.index))+0.5, x.index, rotation='vertical')
        # plt.yticks(np.arange(len(x.columns))+0.5, x.columns)
        # plt.gca().invert_yaxis()

        ## changes the color of rectangle for autocorrelation heatmaps ---------------------------

        # if self._call == 'auto_corr':
        #     for idx in np.arange(len(x.index)):
        #         plt.gca().add_patch(
        #             Rectangle((idx, idx), 1, 1, fill=False, edgecolor='red')
        #         )

        ## end ------------------------------------------------------------------------------------

        ## annotation
        # for idx_row, row in enumerate(x.index):
        #     for idx_col, col in enumerate(x.columns):

        #         if self._call in ['auto_corr', 'cross_corr', 'factor_analysis']:

        #             if abs(x.at[row, col]) > x.values.max() / 2.0:
        #                 color = 'white'
        #             else:
        #                 color = 'black'

        #             plt.text(
        #                 idx_row + 0.5,
        #                 idx_col + 0.5,
        #                 "{:3.2f}".format(x.at[row, col]),
        #                 ha="center",
        #                 va="center",
        #                 color=color)

        #         else:
        #             if x.at[row, col] > 0:

        #                 if x.at[row, col] > x.values.max() / 2.0:
        #                     color = 'white'
        #                 else:
        #                     color = 'black'

        #                 plt.text(
        #                     idx_row + 0.5,
        #                     idx_col + 0.5,
        #                     int(x.at[row, col]),
        #                     ha="center",
        #                     va="center",
        #                     color=color)

        # plt.tight_layout()
        # plt.show()
