#     #----------------------------------------------------------------------------------------------
#     def altair_barhplot(self, color='Greys'):
#         """


#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_year().altair_barhplot()
#         alt.Chart(...)

#         .. image:: ../figs/altair_barhplot.jpg
#             :width: 800px
#             :align: center

#         """
#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         columns = self.columns.tolist()
#         data = pd.DataFrame(self.copy())
#         if data.columns[1] != 'Cited by':
#             data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
#             data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))
#         if columns[0] == 'Year':
#             data = data.sort_values(by=columns[0], ascending=False)
#         return alt.Chart(data).mark_bar().encode(
#             alt.Y(columns[0] + ':N', sort=alt.EncodingSortField(
#                 field=columns[1] + ':Q')),
#             alt.X(columns[1] + ':Q'),
#             alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme=color)))

#     #----------------------------------------------------------------------------------------------
#     def altair_barplot(self):
#         """Vertical bar plot in Altair.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_year().altair_barplot()
#         alt.Chart(...)

#         .. image:: ../figs/altair_barplot.jpg
#             :width: 500px
#             :align: center
#         """
#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         columns = self.columns.tolist()
#         data = pd.DataFrame(self.copy())
#         data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
#         data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

#         return alt.Chart(data).mark_bar().encode(
#             alt.X(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
#             alt.Y(columns[1] + ':Q'),
#             alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys')))

#     #----------------------------------------------------------------------------------------------------
#     def altair_circle(self, ascending_r=None, ascending_c=None, filename=None, **kwds):
#         """Altair scatter plot with filled circles for visualizing relationships.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.auto_corr(
#         ...     column='Authors',
#         ...     sep=',',
#         ...     top_n=30
#         ... ).altair_circle()
#         alt.Chart(...)

#         .. image:: ../figs/altair_circle.png
#             :width: 800px
#             :align: center

#         """
#         if len(self.columns) != 4:
#             Exception('Invalid call for result of function:' + self._call)

#         if ascending_r is None or ascending_r is True:
#             sort_X = 'ascending'
#         else:
#             sort_X = 'descending'

#         if ascending_c is None or ascending_c is True:
#             sort_Y = 'ascending'
#         else:
#             sort_Y = 'descending'

#         chart = alt.Chart(self).mark_circle().encode(
#             alt.X(self.columns[0] + ':N',
#                 axis=alt.Axis(labelAngle=270),
#                 sort=sort_X),
#             alt.Y(self.columns[1] + ':N',
#                 sort=sort_Y),
#             size=self.columns[2],
#             color=self.columns[2])

#         if filename is not None:
#             char.save(filename)

#         return chart


#     #----------------------------------------------------------------------------------------------------
#     def altair_heatmap(self, ascending_r=None, ascending_c=None, filename=None, **kwds):
#         """Altair Heatmap
#         Available cmaps:

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.terms_by_year(
#         ...    column='Authors',
#         ...    sep=',',
#         ...    top_n=20).altair_heatmap()
#         alt.Chart(...)

#         .. image:: ../figs/altair_heatmap.jpg
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

#         _self = self.copy()
#         _self[_self.columns[0]] = _self[_self.columns[0]].map(lambda w: cut_text(w))
#         _self[_self.columns[1]] = _self[_self.columns[1]].map(lambda w: cut_text(w))

#         if ascending_r is None or ascending_r is True:
#             sort_X = 'ascending'
#         else:
#             sort_X = 'descending'

#         if ascending_c is None or ascending_c is True:
#             sort_Y = 'ascending'
#         else:
#             sort_Y = 'descending'

#         graph = alt.Chart(_self).mark_rect().encode(
#             alt.X(_self.columns[0] + ':O', sort=sort_X),
#             alt.Y(_self.columns[1] + ':O', sort=sort_Y),
#             color=_self.columns[2] + ':Q')

#         if self._call == 'co_ocurrence':
#             text = graph.mark_text(
#                 align='center',
#                 baseline='middle',
#                 dx=5
#             ).encode(
#                 text=_self.columns[2] + ':Q'
#             )
#         else:
#             text = None

#         plt.tight_layout()

#         return graph
