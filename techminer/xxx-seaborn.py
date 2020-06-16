"""
Plots using Seaborn
==================================================================================================




"""

#     #----------------------------------------------------------------------------------------------
#     def seaborn_barhplot(self, color='gray'):
#         """

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_year().seaborn_barhplot()


#         .. image:: ../figs/seaborn_barhplot.jpg
#             :width: 600px
#             :align: center

#         """
#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         columns = self.columns.tolist()
#         data = pd.DataFrame(self.copy())
#         if data.columns[1] != "Times_Cited":
#             data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
#             data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

#         if columns[0] == 'Year':
#             data = data.sort_values(by=columns[0], ascending=False)
#         else:
#             data = data.sort_values(by=columns[1], ascending=False)
#         sns.barplot(
#             x=columns[1],
#             y=columns[0],
#             data=data,
#             label=columns[0],
#             color=color)
#         plt.gca().xaxis.grid(True)

#     #----------------------------------------------------------------------------------------------
#     def seaborn_barplot(self, color='gray'):
#         """Vertical bar plot in Seaborn.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_year().seaborn_barplot()

#         .. image:: ../figs/seaborn_barhplot.jpg
#             :width: 800px
#             :align: center
#         """
#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         columns = self.columns.tolist()
#         data = Result(self.copy())
#         data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
#         data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

#         columns = data.columns.tolist()
#         result = sns.barplot(
#             y=columns[1],
#             x=columns[0],
#             data=data,
#             label=columns[0],
#             color=color)
#         _, labels = plt.xticks()
#         result.set_xticklabels(labels, rotation=90)
#         plt.gca().yaxis.grid(True)

#     #----------------------------------------------------------------------------------------------------
#     def seaborn_heatmap(self, ascending_r=None, ascending_c=None, filename=None):
#         """Heat map.


#         https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'


#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.terms_by_year(
#         ...    column='Authors',
#         ...    sep=',',
#         ...    top_n=20).seaborn_heatmap()
#         >>> plt.savefig('./figs/seaborn_heatmap.jpg')

#         .. image:: ../figs//seaborn_heatmap.jpg
#             :width: 600px
#             :align: center

#         """

#         if len(self.columns) != 4:
#             Exception('Invalid call for result of function:' + self._call)

#         ## force the same order of cells in rows and cols ------------------------------------------
#         if self._call == 'auto_corr':
#             if ascending_r is None and ascending_c is None:
#                 ascending_r = True
#                 ascending_c = True
#             elif ascending_r is not None and ascending_r != ascending_c:
#                 ascending_c = ascending_r
#             elif ascending_c is not None and ascending_c != ascending_r:
#                 ascending_r = ascending_c
#             else:
#                 pass
#         ## end -------------------------------------------------------------------------------------


#         sns.set()
#         _self = self.tomatrix(ascending_r, ascending_c)
#         _self = _self.transpose()
#         _self.columns = [cut_text(w) for w in _self.columns]
#         _self.index = [cut_text(w) for w in _self.index]

#         sns_plot = sns.heatmap(_self)

#         if filename is not None:
#             sns_plot.savefig(filename)

#         #return sns_plot


#     #----------------------------------------------------------------------------------------------------
#     def seaborn_relplot(self, ascending_r=None, ascending_c=None, filename=None):
#         """Seaborn relplot plot with filled circles for visualizing relationships.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.auto_corr(
#         ...     column='Authors',
#         ...     sep=',',
#         ...     top_n=30
#         ... ).seaborn_relplot(filename='./figs/seaborn_relplot.png')

#         .. image:: ../figs//seaborn_relplot.png
#             :width: 600px
#             :align: center
#         """

#         if len(self.columns) != 4:
#             Exception('Invalid call for result of function:' + self._call)

#         sns_plot = sns.relplot(
#             x = self.columns[0],
#             y = self.columns[1],
#             size = self.columns[2],
#             alpha = 0.8,
#             palette = 'viridis',
#             data = self)
#         plt.xticks(rotation=90)
#         if filename is not None:
#             sns_plot.savefig(filename)
