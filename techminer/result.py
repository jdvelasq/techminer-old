"""
TechMiner.Result
==================================================================================================

"""
# import altair as alt
# import geopandas
# import geoplot
# import itertools
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from techminer.common import *
# from collections import OrderedDict
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.optimize import minimize
# from shapely.geometry import Point, LineString
# from sklearn.cluster import KMeans
# from matplotlib.patches import Rectangle
# from wordcloud import WordCloud, ImageColorGenerator

# #----------------------------------------------------------------------------------------------------
# def _compute_graph_layout(graph):

#     path_length = nx.shortest_path_length(graph)
#     distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
#     for row, data in path_length:
#         for col, dist in data.items():
#             distances.loc[row,col] = dist
#     distances = distances.fillna(distances.max().max())

#     return nx.kamada_kawai_layout(graph, dist=distances.to_dict())

# #--------------------------------------------------------------------------------------------------------
# class Result(pd.DataFrame):
#     """Class implementing a dataframe with results of analysis.
#     """
#     #----------------------------------------------------------------------------------------------------
#     def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
#             cluster_data=None, call=None):

#         super().__init__(data, index, columns, dtype, copy)
#         self._call = call
#         self._cluster_data = None
#         self._cluster_data = cluster_data

#     #----------------------------------------------------------------------------------------------------
#     @property
#     def _constructor_expanddim(self):
#         return self


#     #----------------------------------------------------------------------------------------------
#     def _add_count_to_label(self, column):

#         count = self.groupby(by=column, as_index=True)[self.columns[-2]].sum()
#         count = {key : value for key, value in zip(count.index, count.tolist())}
#         self[column] = self[column].map(lambda x: cut_text(str(x) + ' [' + str(count[x]) + ']'))

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

#     #----------------------------------------------------------------------------------------------
#     def barhplot(self, color='gray', figsize=(12,8)):
#         """Plots a pandas.DataFrame using Altair.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_year().barhplot()

#         .. image:: ../figs/barhplot.jpg
#             :width: 600px
#             :align: center
#         """
#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         data = pd.DataFrame(self.copy())
#         columns = data.columns.tolist()
#         if data.columns[1] != 'Cited by':
#             data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
#             data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

#         if columns[0] == 'Year':
#             data =  data.sort_values(by=columns[0], ascending=True)
#         else:
#             data =  data.sort_values(by=columns[1], ascending=True)

#         #plt.figure(figsize=figsize)
#         data.plot.barh(columns[0], columns[1], color=color, figsize=figsize)
#         plt.gca().xaxis.grid(True)


#     #----------------------------------------------------------------------------------------------
#     def barplot(self, color='gray', figsize=(8,12)):
#         """Vertical bar plot in matplotlib.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_year().barplot()

#         .. image:: ../figs/barplot.jpg
#             :width: 600px
#             :align: center

#         """
#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         columns = self.columns.tolist()

#         plt.figure(figsize=figsize)
#         data = pd.DataFrame(self.copy())
#         data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
#         data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))
#         data.plot.bar(columns[0], columns[1], color=color)
#         plt.gca().yaxis.grid(True)


#     #----------------------------------------------------------------------------------------------------
#     def chord_diagram(self, figsize=(12, 12), minval=None, R=3, n_bezier=100, dist=0.2):
#         """Creates a chord diagram for representing clusters.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.auto_corr(
#         ...     column='Authors',
#         ...     sep=',',
#         ...     top_n=20).chord_diagram()
#         >>> plt.savefig('./figs/chord-diagram.jpg')

#         .. image:: ../figs/chord-diagram.jpg
#             :width: 800px
#             :align: center

#         """

#         if  self._cluster_data is None:
#             Exception('Invalid call for result of function:' + self._call)

#         chord_diagram(
#             self[self.columns[0]].unique(),
#             self._cluster_data,
#             figsize=figsize,
#             minval=minval,
#             R=R,
#             n_bezier=n_bezier,
#             dist=dist)


#     #----------------------------------------------------------------------------------------------------
#     def heatmap(self, ascending_r=None, ascending_c=None, figsize=(10, 10), cmap='Blues'):
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
#         ...    top_n=20).heatmap(figsize=(8,4))
#         >>> plt.savefig('./figs/heatmap.jpg')

#         .. image:: ../figs//heatmap.jpg
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


#         x = self.tomatrix(ascending_r, ascending_c)

#         ## rename columns and row index
#         x.columns = [cut_text(w) for w in x.columns]
#         x.index = [cut_text(w) for w in x.index]

#         plt.figure(figsize=figsize)

#         if self._call == 'factor_analysis':
#             x = self.tomatrix(ascending_r, ascending_c)
#             x = x.transpose()
#             ## x = x.apply(lambda w: abs(w))
#             plt.pcolor(np.transpose(abs(x.values)), cmap=cmap)
#         else:
#             plt.pcolor(np.transpose(x.values), cmap=cmap)

#         #plt.pcolor(np.transpose(x.values), cmap=cmap)
#         plt.xticks(np.arange(len(x.index))+0.5, x.index, rotation='vertical')
#         plt.yticks(np.arange(len(x.columns))+0.5, x.columns)
#         ## plt.gca().set_aspect('equal', 'box')
#         plt.gca().invert_yaxis()

#         ## changes the color of rectangle for autocorrelation heatmaps ---------------------------

#         # if self._call == 'auto_corr':
#         #     for idx in np.arange(len(x.index)):
#         #         plt.gca().add_patch(
#         #             Rectangle((idx, idx), 1, 1, fill=False, edgecolor='red')
#         #         )

#         ## end ------------------------------------------------------------------------------------


#         ## annotation
#         for idx_row, row in enumerate(x.index):
#             for idx_col, col in enumerate(x.columns):

#                 if self._call in ['auto_corr', 'cross_corr', 'factor_analysis']:

#                     if abs(x.at[row, col]) > x.values.max() / 2.0:
#                         color = 'white'
#                     else:
#                         color = 'black'

#                     plt.text(
#                         idx_row + 0.5,
#                         idx_col + 0.5,
#                         "{:3.2f}".format(x.at[row, col]),
#                         ha="center",
#                         va="center",
#                         color=color)

#                 else:
#                     if x.at[row, col] > 0:

#                         if x.at[row, col] > x.values.max() / 2.0:
#                             color = 'white'
#                         else:
#                             color = 'black'

#                         plt.text(
#                             idx_row + 0.5,
#                             idx_col + 0.5,
#                             int(x.at[row, col]),
#                             ha="center",
#                             va="center",
#                             color=color)


#         plt.tight_layout()
#         plt.show()


#     #----------------------------------------------------------------------------------------------------
#     def map(self, min_value=None, top_links=None, figsize = (10,10),
#             font_size=12, factor=None, size=(25,300)):
#         """
#         Draw an autocorrelation, crosscorrelation or factor map.


#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.auto_corr(
#         ...     column='Authors',
#         ...     sep=',',
#         ...     top_n=20).map()
#         >>> plt.savefig('./figs/autocorr-map.jpg')

#         .. image:: ../figs/autocorr-map.jpg
#             :width: 800px
#             :align: center

#         """

#         if self._cluster_data is None:
#             Exception('Invalid call for result of function:' + self._call)

#         ## cluster dataset
#         cluster_data = self._cluster_data.copy()

#         ## figure properties
#         plt.figure(figsize=figsize)

#         ## graph
#         graph = nx.Graph()

#         ## adds nodes to graph
#         clusters = list(set(cluster_data.cluster))
#         nodes = list(set(self.tomatrix().index))

#         graph.add_nodes_from(clusters)
#         graph.add_nodes_from(nodes)


#         ## adds edges and properties
#         weigth = []
#         style = []
#         value = []
#         for _, row in cluster_data.iterrows():
#             graph.add_edge(row[1], row[2])
#             if row[3] >= 0.75:
#                 weigth += [4]
#                 style += ['solid']
#                 value += [row[3]]
#             elif row[3] >= 0.50:
#                 weigth += [2]
#                 style += ['solid']
#                 value += [row[3]]
#             elif row[3] >= 0.25:
#                 weigth += [1]
#                 style += ['dashed']
#                 value += [row[3]]
#             else:
#                 weigth += [1]
#                 style += ['dotted']
#                 value += [row[3]]


#         edges = pd.DataFrame({
#             'edges' : graph.edges(),
#             'weight' : weigth,
#             'style' : style,
#             'value' : value
#         })

#         edges = edges.sort_values(by='value', ascending=False)

#         if top_links is not None and top_links < len(edges):
#                 edges = edges[0:top_links]

#         if min_value is not None:
#             edges = edges[edges['value'] >= min_value]

#         ## edges from center of cluster to nodes.
#         for _, row in cluster_data.iterrows():
#             graph.add_edge(row[0], row[1])
#             graph.add_edge(row[0], row[2])


#         ## graph layout
#         path_length = nx.shortest_path_length(graph)
#         distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
#         for row, data in path_length:
#             for col, dist in data.items():
#                 distances.loc[row,col] = dist
#         distances = distances.fillna(distances.max().max())
#         layout = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

#         ## nodes drawing
#         node_size = [x[(x.find('[')+1):-1] for x in nodes]
#         node_size = [float(x) for x in node_size]
#         max_node_size = max(node_size)
#         min_node_size = min(node_size)
#         node_size = [size[0] + x / (max_node_size - min_node_size) * size[1] for x in node_size]

#         nx.draw_networkx_nodes(
#             graph,
#             layout,
#             nodelist=nodes,
#             node_size=node_size,
#             node_color='red')

#         ## edges drawing
#         for style in list(set(edges['style'].tolist())):

#             edges_set = edges[edges['style'] == style]

#             if len(edges_set) == 0:
#                 continue

#             nx.draw_networkx_edges(
#                 graph,
#                 layout,
#                 edgelist=edges_set['edges'].tolist(),
#                 style=style,
#                 width=edges_set['weight'].tolist(),
#                 edge_color='black')


#         ## node labels
#         x_left, x_right = plt.xlim()
#         y_left, y_right = plt.ylim()
#         delta_x = (x_right - x_left) * 0.01
#         delta_y = (y_right - y_left) * 0.01
#         for node in nodes:
#             x_pos, y_pos = layout[node]
#             plt.text(
#                 x_pos + delta_x,
#                 y_pos + delta_y,
#                 node,
#                 size=font_size,
#                 ha='left',
#                 va='bottom',
#                 bbox=dict(
#                     boxstyle="square",
#                     ec='lightgray',
#                     fc='white',
#                     ))

#         if factor is not None:
#             left, right = plt.xlim()
#             width = (right - left) * factor / 2.0
#             plt.xlim(left - width, right + width)

#         plt.axis('off')

#     #----------------------------------------------------------------------------------------------------
#     def ocurrence_map(self, min_value=None, top_links=None, figsize = (10,10),
#             font_size=12, factor=None, size=(300,1000)):
#         """Cluster map for ocurrence and co-ocurrence matrices.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.co_ocurrence(
#         ...    column_r='Authors',
#         ...    column_c='Authors',
#         ...    sep_r=',',
#         ...    sep_c=',',
#         ...    top_n=10
#         ... ).heatmap()
#         >>> plt.savefig('./figs/heatmap-ocurrence-map.jpg')

#         .. image:: ../figs/heatmap-ocurrence-map.jpg
#             :width: 600px
#             :align: center

#         >>> rdf.co_ocurrence(
#         ...    column_r='Authors',
#         ...    column_c='Authors',
#         ...    sep_r=',',
#         ...    sep_c=',',
#         ...    top_n=10
#         ... ).ocurrence_map(
#         ...    figsize=(11,11),
#         ...    font_size=10,
#         ...    factor = 0.1,
#         ...    size=(300,1000)
#         ... )
#         >>> plt.savefig('./figs/ocurrence-map.jpg')

#         .. image:: ../figs/ocurrence-map.jpg
#             :width: 600px
#             :align: center

#         """

#         if self._call not in  ['ocurrence', 'co_ocurrence']:
#             Exception('Invalid call for result of function:' + self._call)


#         ## figure properties
#         plt.figure(figsize=figsize)

#         ## graph
#         graph = nx.Graph()

#         terms_r = list(set(self.tomatrix().index.tolist()))
#         terms_c = list(set(self.tomatrix().columns.tolist()))

#         nodes = list(set(terms_r + terms_c))
#         nodes = [cut_text(x) for x in nodes]
#         graph.add_nodes_from(nodes)

#         if sorted(terms_r) != sorted(terms_c):

#             numnodes = [str(i) for i in range(len(self))]
#             graph.add_nodes_from(numnodes)

#             for idx, row in self.iterrows():
#                 graph.add_edge(row[0], str(idx))
#                 graph.add_edge(row[1], str(idx))

#             labels={str(idx):row[2] for idx, row in self.iterrows()}

#         else:

#             mtx = self.tomatrix()
#             edges = []
#             labels = {}

#             n = 0
#             for idx_r, row in enumerate(mtx.index.tolist()):
#                 for idx_c, col in enumerate(mtx.columns.tolist()):

#                     if idx_c < idx_r:
#                         continue

#                     if mtx.at[row, col] > 0:
#                         edges += [(row, str(n)), (col, str(n))]
#                         labels[str(n)] = mtx.at[row, col]
#                         n += 1


#             numnodes = [str(i) for i in range(n)]
#             graph.add_nodes_from(numnodes)

#             for a, b in edges:
#                 graph.add_edge(a, b)

#         ## graph layout
#         layout = _compute_graph_layout(graph)

#         ## draw terms nodes
#         node_size = [int(n[n.find('[')+1:-1])  for n in nodes]
#         node_size = [size[0] + (n - min(node_size)) / (max(node_size) - min(node_size)) * (size[1] - size[0]) for n in node_size]
#         nx.draw_networkx_nodes(
#             graph,
#             layout,
#             nodelist=nodes,
#             node_size=node_size,
#             node_color='red')

#         x_left, x_right = plt.xlim()
#         y_left, y_right = plt.ylim()
#         delta_x = (x_right - x_left) * 0.01
#         delta_y = (y_right - y_left) * 0.01
#         for node in nodes:
#             x_pos, y_pos = layout[node]
#             plt.text(
#                 x_pos + delta_x,
#                 y_pos + delta_y,
#                 node,
#                 size=font_size,
#                 ha='left',
#                 va='bottom',
#                 bbox=dict(
#                     boxstyle="square",
#                     ec='gray',
#                     fc='white',
#                     ))

#         # nx.draw_networkx_labels(
#         #     graph,
#         #     layout,
#         #     labels={t:t for t in terms},
#         #     bbox=dict(facecolor='none', edgecolor='lightgray', boxstyle='round'))

#         ## draw quantity nodes
#         node_size = [int(labels[n]) for n in labels.keys()]
#         node_size = [size[0] + (n - min(node_size)) / (max(node_size) - min(node_size)) * (size[1] - size[0]) for n in node_size]
#         nx.draw_networkx_nodes(
#             graph,
#             layout,
#             nodelist=numnodes,
#             node_size=node_size,
#             node_color='lightblue')

#         nx.draw_networkx_labels(
#             graph,
#             layout,
#             labels=labels,
#             font_color='black')

#         ## edges
#         nx.draw_networkx_edges(
#             graph,
#             layout,
#             width=1
#         )
#         plt.axis('off')


#     #----------------------------------------------------------------------------------------------------
#     def print_IDs(self):
#         """Auxiliary function to print IDs of documents.
#         """

#         if self._call in ['co_ocurrence', 'cross_corr', 'auto_corr']:

#             for idx, row in self.iterrows():
#                 if row[-1] is not None:
#                     print(row[0], ', ', row[1], ' (', len(row[-1]), ')', ' : ', sep='', end='')
#                     for i in row[-1]:
#                         print(i, sep='', end='')
#                     print()

#         elif self._call == 'terms_by_terms_by_year':

#             for idx, row in self.iterrows():
#                 if row[-1] is not None:
#                     print(row[0], ', ', row[1], ', ', row[2], ' (', len(row[-1]), ')', ' : ', sep='', end='')
#                     for i in row[-1]:
#                         print(i, sep='', end='')
#                     print()

#         elif self._call == 'factor_analysis':
#             pass
#         else:
#             pass


#     #----------------------------------------------------------------------------------------------------
#     def sankey_plot(self, figsize=(7,10), minval=None):
#         """Cross-relation sankey plot.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.cross_corr(
#         ...    column_r='keywords (cleaned)',
#         ...    sep_r=';',
#         ...    column_c='Authors',
#         ...    sep_c=','
#         ... ).sankey_plot(minval=0.1)
#         >>> plt.savefig('./figs/sankey-plot.jpg')

#         .. image:: ../figs//sankey-plot.jpg
#             :width: 600px
#             :align: center


#         """
#         if self._call != 'cross_corr':
#             Exception('Invalid call for result of function:' + self._call)

#         x = self

#         llabels = sorted(list(set(x[x.columns[0]])))
#         rlabels = sorted(list(set(x[x.columns[1]])))

#         factorL = max(len(llabels)-1, len(rlabels)-1) / (len(llabels) - 1)
#         factorR = max(len(llabels)-1, len(rlabels)-1) / (len(rlabels) - 1)

#         lpos = {k:v*factorL for v, k in enumerate(llabels)}
#         rpos = {k:v*factorR for v, k in enumerate(rlabels)}

#         fig, ax1 = plt.subplots(figsize=(7, 10))
#         ax1.scatter([0] * len(llabels), llabels, color='black', s=50)

#         for index, r in x.iterrows():

#             row = r[0]
#             col = r[1]
#             val = r[2]

#             if val >= 0.75:
#                 linewidth = 4
#                 linestyle = '-'
#             elif val >= 0.50:
#                 linewidth = 2
#                 linstyle = '-'
#             elif val >= 0.25:
#                 linewidth = 2
#                 linestyle = '--'
#             elif val < 0.25:
#                 linewidth = 1
#                 linestyle = ':'
#             else:
#                 linewidth = 0
#                 linestyle = '-'

#             if minval is  None:
#                 plt.plot(
#                     [0, 1],
#                     [lpos[row], rpos[col]],
#                     linewidth=linewidth,
#                     linestyle=linestyle,
#                     color='black')
#             elif abs(val) >= minval :
#                 plt.plot(
#                     [0, 1],
#                     [lpos[row], rpos[col]],
#                     linewidth=linewidth,
#                     linestyle=linestyle,
#                     color='black')

#         ax2 = ax1.twinx()
#         ax2.scatter([1] * len(rlabels), rlabels, color='black', s=50)
#         #ax2.set_ylim(0, len(rlabels)-1)


#         for txt in ['bottom', 'top', 'left', 'right']:
#             ax1.spines[txt].set_color('white')
#             ax2.spines[txt].set_color('white')

#         ax2.set_xticks([])

#         plt.tight_layout()

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
#         if data.columns[1] != 'Cited by':
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


#     #----------------------------------------------------------------------------------------------------
#     def tomatrix(self, ascending_r=None, ascending_c=None):
#         """Displays a term by term dataframe as a matrix.

#         >>> mtx = Result({
#         ...   'rows':['r0', 'r1', 'r2', 'r0', 'r1', 'r2'],
#         ...   'cols':['c0', 'c1', 'c0', 'c1', 'c0', 'c1'],
#         ...   'vals':[ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0]
#         ... })
#         >>> mtx
#           rows cols  vals
#         0   r0   c0   1.0
#         1   r1   c1   2.0
#         2   r2   c0   3.0
#         3   r0   c1   4.0
#         4   r1   c0   5.0
#         5   r2   c1   6.0

#         >>> mtx.tomatrix() # doctest: +NORMALIZE_WHITESPACE
#              c0   c1
#         r0  1.0  4.0
#         r1  5.0  2.0
#         r2  3.0  6.0

#         """

#         # if self._call not in [
#         #     'coo-matrix',
#         #     'cross-matrix',
#         #     'auto-matrix']:

#         #     raise Exception('Invalid function call for type: ' + self._call )


#         if self.columns[0] == 'Year':
#             year = self.Year.copy()
#             dict_year = { x[0:x.find(' [')] : x for x in year}
#             year = year.map(lambda x: int(x[0:x.find('[')]))
#             year = [str(x) for x in range(min(year), max(year)+1)]
#             year = [y + ' [0]' if y not in dict_year.keys() else dict_year[y]  for y in year]
#             termA_unique = year
#             # termA_unique = range(min(self.Year), max(self.Year)+1)
#         else:
#             termA_unique = self.iloc[:,0].unique()

#         if self.columns[1] == 'Year':
#             year = self.Year.copy()
#             dict_year = {x[0:x.find(' [')] : x   for x in year}
#             year = year.map(lambda x: int(x[0:x.find('[')]))
#             year = [str(x) for x in range(min(year), max(year)+1)]
#             year = [y + ' [0]' if y not in dict_year.keys() else dict_year[y]  for y in year]
#             termB_unique = year
#             # termB_unique = range(min(self.Year), max(self.Year)+1)
#         else:
#             termB_unique = self.iloc[:,1].unique()

#         if ascending_r is not None:
#             termA_unique = sorted(termA_unique, reverse = not ascending_r)

#         if ascending_c is not None:
#             termB_unique = sorted(termB_unique, reverse = not ascending_c)

#         if self._call == 'co_ocurrence':
#             result = pd.DataFrame(
#                 np.full((len(termA_unique), len(termB_unique)), 0)
#             )

#         else:
#             result = pd.DataFrame(
#                 np.zeros((len(termA_unique), len(termB_unique)))
#             )

#         result.columns = termB_unique
#         result.index = termA_unique

#         for index, r in self.iterrows():
#             row = r[0]
#             col = r[1]
#             val = r[2]
#             result.loc[row, col] = val

#         return Result(result, call='Matrix')

#     #----------------------------------------------------------------------------------------------------
#     def transpose(self, *args, **kwargs):
#         """Transpose results matrix.
#         """
#         return Result(super().transpose(), call=self._call)


#     #----------------------------------------------------------------------------------------------------
#     #TODO personalizar valor superior para escalar los pesos de los puentes
#     #TODO map
#     def network(self, save=False, name='network.png', corr_min=0.7, node_color='lightblue',
#                   edge_color='lightgrey', edge_color2='lightcoral', node_size=None, fond_size=4,
#                   figsize = (10,10)):
#         """
#         This function generates network graph for matrix.

#         Args:
#             matrix (pandas.DataFrame): Matrix with variables on indexes and column titles
#             save (boolean): If True, the graph will save with the name given
#             name (str): Name to save the png file with the image
#             corr_min (int): Minimum absolute value for  the relationships between variables
#                             to be shown in the graph.
#                             It is suggested when a correlation matrix is ​​being used
#             node_color (str): Color name used to plot nodes
#             edge_color (str): Color name used to plot edges with positive weights
#             edge_color2 (str): Color name used to plot edges with negative weights
#             node_size (int): If None value, the size of the nodes is plotted according
#                             to the weights of edges that arrive and leave each one of them.
#                             If numeric value, all nodes will be plotted with this given size
#             fond_size (int): Node label fond size
#             figsize (float, float): size of figure drawn

#         Returns:
#             None


#         """

#         if self._call not in [
#             'co_ocurrence',
#             'cross_corr',
#             'auto_corr',
#             'factor_analysis']:

#             raise Exception('Invalid function call for type: ' + self._call )


#         if self._call == 'factor_analysis':
#             x = self.copy()
#         else:
#             x = self.tomatrix()

#         plt.clf()
#         plt.figure(figsize=figsize)

#         #generate network graph
#         graph = nx.Graph()
#         # add nodes
#         rows = x.index
#         columns = x.columns
#         nodes = list(set(rows.append(columns)))

#         #add nodes
#         graph.add_nodes_from(nodes)
#         list_ = list(OrderedDict.fromkeys(itertools.product(rows, columns)))
#         if len(rows) == len(columns) and (all(rows.sort_values())==all(columns.sort_values())):
#             list_ = list(set(tuple(sorted(t)) for t in list_))

#         # add edges
#         for i in range(len(list_)):
#             combinations=list_[i]
#             from_node, to_node = combinations[0], combinations[1]
#             if from_node != to_node:
#                 weight = x.loc[from_node, to_node]
#                 if weight != 0 and abs(weight)>corr_min:
#                     if weight<0:
#                         weight=abs(weight)
#                         edge_colour =edge_color2
#                     else:
#                         edge_colour = edge_color
#                     graph.add_edge(from_node, to_node, weight=weight, color = edge_colour)

#         #calculate distance between relationated nodes to avoid overlaping
#         path_length = nx.shortest_path_length(graph)
#         distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
#         for row, data in path_length:
#             for col, dist in data.items():
#                 distances.loc[row,col] = dist
#         distances = distances.fillna(distances.max().max() )

#         #layout of graph
#         pos = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

#         #weights and colors of the relationships between nodes for edges thickness
#         weights = dict(((u, v), int(d["weight"])) for u, v, d in graph.edges(data=True))
#         colors = dict(((u, v), d["color"]) for u, v, d in graph.edges(data=True))

#         #Edges weights for plot
#         max_=max([i for i in weights.values()])
#         min_=min([i for i in weights.values()])
#         min_range=1
#         max_range=5
#         if max_<=1:
#             width = ([(1+x)*2 for x in weights.values()])
#         else:
#             width = ([((((x-min_)/(max_-min_))*(max_range-min_range))+min_range) for x in weights.values()])
#             # width=list(weights.values())

#         #node sizes
#         if not node_size:
#             node_sizes = dict(graph.degree())
#             node_sizes = ([(x)*10 for key,x in node_sizes.items()])
#         else:
#             node_sizes=node_size

#         #visual graph configuration
#         nx.draw(graph, pos,node_size=node_sizes, node_color=node_color,
#                 edge_color=list(colors.values()), font_size=fond_size,
#                 with_labels=True, width=width)

#         #save figure as png
#         if save:
#             plt.savefig(name, format="PNG", dpi=300, bbox_inches='tight')

#         plt.tight_layout()
#         plt.show()
#         return None


#     #----------------------------------------------------------------------------------------------------
#     #TODO networkmap validar como pasar lonlat,
#     #que pasa si valores negativos???
#     #personalizar tamaño de la figura,
#     #guardar archivo
#     #quitar ejes

#     def networkmap(matrix, color_edges ='grey', color_node='red',color_map = 'white', edge_map = 'lightgrey', node_size =None, edge_weight = None):

#         """
#         This function generates network graph over map, for matrix with country relations.

#         Args:
#             matrix (pandas.DataFrame): Matrix with variables on indexes and column titles
#             color_edges (str): Color name used to plot edges
#             color_node (str): Color name used to plot nodes
#             color_map (str): Color name used to plot map countries
#             edge_map (str): Color name used to plot contries border
#             node_size (int): If None value, the size of the nodes is plotted according
#                             to the weights of edges that arrive and leave each one of them.
#                             If numeric value, all nodes will be plotted with this given size
#             edge_weight (int): If None value, the weigth of the edges is plotted according
#                             to matrix values
#                             If numeric value, all edges will be plotted with this given size
#         Returns:
#             None
#         #
#         """

#         #Get longitudes and latituds
#         lonlat=pd.read_csv('LonLat.csv',sep=';')

#         #node's names
#         rows=matrix.index
#         columns=matrix.columns
#         nodes=list(set(rows.append(columns)))
#         nodes = [row.replace(' ', '') for row in rows ]


#         #nodes_combinations
#         list_ = list(OrderedDict.fromkeys(itertools.product(rows, columns)))
#         if len(rows)== len(columns) and (all(rows.sort_values())==all(columns.sort_values())):
#             list_=list(set(tuple(sorted(t)) for t in list_))


#         pos=lonlat[lonlat.country.isin(nodes)]

#         geometry = [Point(xy) for xy in zip(pos['lon'], pos['lat'])]

#         # Coordinate reference system : WGS84
#         crs = {'init': 'epsg:4326'}

#         # Creating a Geographic data frame from nodes
#         gdf = geopandas.GeoDataFrame(pos, crs=crs, geometry=geometry)

#         #edges
#         df=pd.DataFrame({'initial':[],'final':[],'initial_lon': [], 'initial_lat': [],'final_lon': [],'final_lat': [], 'weight': []})
#         for i in range(len(list_)):
#             combinations=list_[i]
#             from_node, to_node = combinations[0],combinations[1]
#             if from_node != to_node:
#                 weight =matrix.loc[from_node,to_node]
#                 if weight != 0:
#                     df = df.append({'initial':from_node.replace(' ', ''),'final':to_node.replace(' ', ''),'initial_lon': pos[pos.country==from_node.replace(' ', '')]['lon'].values, 'initial_lat': pos[pos.country==from_node.replace(' ', '')]['lat'].values,'final_lon': pos[pos.country==to_node.replace(' ', '')]['lon'].values,'final_lat': pos[pos.country==to_node.replace(' ', '')]['lat'].values, 'weight': weight}, ignore_index='True')

#         # Creating a Geographic data frame from edges
#         df['orig_coord'] = [Point(xy) for xy in zip(df['initial_lon'], df['initial_lat'])]
#         df['dest_coord'] = [Point(xy) for xy in zip(df['final_lon'], df['final_lat'])]

#         geometry_lines=[LineString(xy) for xy in zip(df.orig_coord,df.dest_coord)]
#         gdf_lines=geopandas.GeoDataFrame(df, crs=crs, geometry=geometry_lines)

#         #base map
#         world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

#         #nodes size
#         if not node_size:
#             nodes_freq=list(gdf_lines.initial) + list(gdf_lines.final)
#             nodes_freq.sort()
#             nodes_size= {x:nodes_freq.count(x) for x in nodes_freq}
#             size=nodes_size.values()
#             size=[x*5 for x in size]
#         else:
#             size=node_size

#         #edges weigth
#         if not node_size:
#             edges_=list(gdf_lines.weight)
#         else:
#             edges_= node_size
#         #plot graph
#         gdf.plot(ax=world.plot(ax=gdf_lines.plot(color=color_edges, markersize=edges_,alpha=0.5),color=color_map, edgecolor= edge_map), color=color_node,  markersize=size)

#         plt.tight_layout()
#         plt.show()

#         return None

#     #----------------------------------------------------------------------------------------------
#     def wordcloud(self, figsize=(14, 7), max_font_size=50, max_words=100,
#             background_color="white"):
#         """


#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> rdf.documents_by_terms('Source title').wordcloud()

#         .. image:: ../figs/wordcloud.jpg
#             :width: 800px
#             :align: center
#         """

#         if len(self.columns) != 3:
#             Exception('Invalid call for result of function:' + self._call)

#         columns = self.columns.tolist()

#         words = [row[0]  for _, row in self.iterrows() for i in range(row[1])]

#         wordcloud = WordCloud(
#             max_font_size=max_font_size,
#             max_words=max_words,
#             background_color=background_color).generate(' '.join(words))

#         plt.figure(figsize=figsize)
#         plt.imshow(wordcloud, interpolation="bilinear")
#         plt.axis("off")
#         plt.show()

#     #----------------------------------------------------------------------------------------------
#     def worldmap(self, figsize=(14, 7)):
#         """Worldmap plot with the number of documents per country.

#         >>> import pandas as pd
#         >>> import matplotlib.pyplot as plt
#         >>> from techminer.datasets import load_test_cleaned
#         >>> rdf = load_test_cleaned().data
#         >>> from techminer.strings import  *
#         >>> rdf['Country'] = rdf['Affiliations'].map(lambda x: extract_country(x, sep=';'))
#         >>> rdf.documents_by_terms('Country', sep=';').head()
#                   Country  Num Documents                                                 ID
#         0           China             83  [[*3*], [*4*], [*6*], [*6*], [*7*], [*10*], [*...
#         1          Taiwan             20  [[*14*], [*14*], [*17*], [*17*], [*17*], [*17*...
#         2   United States             17  [[*3*], [*22*], [*23*], [*23*], [*26*], [*26*]...
#         3  United Kingdom             15  [[*5*], [*7*], [*11*], [*11*], [*11*], [*28*],...
#         4           India             15  [[*9*], [*50*], [*51*], [*56*], [*56*], [*57*]...
#         >>> rdf.documents_by_terms('Country', sep=';').worldmap()

#         .. image:: ../figs/worldmap.jpg
#             :width: 800px
#             :align: center
#         """

#         if 'Country' not in list(self.columns):
#             raise Exception('No country column found in data')

#         world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#         world = world[world.name != "Antarctica"]
#         world['q'] = 0
#         world.index = world.name

#         rdf = self.copy()
#         rdf['Country'] = rdf['Country'].map(
#             lambda x: x.replace('United States', 'United States of America')
#         )

#         #rdf['Country'] = [w if w !=  else  for w in rdf['Country']]
#         rdf.index = rdf['Country']
#         for country in rdf['Country']:
#             if country in world.index:
#                 world.at[country, 'q'] = rdf.loc[country, 'Num Documents']
#         _, axx = plt.subplots(1, 1, figsize=figsize)
#         divider = make_axes_locatable(axx)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         world.plot(column='q', legend=True, ax=axx, cax=cax, cmap='Pastel2')

#     #----------------------------------------------------------------------------------------------------
