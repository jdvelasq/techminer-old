"""
Sankey plots
==================================================================================================




"""

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
