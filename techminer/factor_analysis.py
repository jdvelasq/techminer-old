"""
Factor analysis
==================================================================================================



"""

# import ipywidgets as widgets
# from IPython.display import HTML, clear_output, display
# from ipywidgets import AppLayout, Layout
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from techminer.plots import COLORMAPS

# from techminer.correlation import compute_tfm
# from techminer.keywords import Keywords
# from techminer.by_term import summary_by_term


# def factor_analysis(x, column, n_components=None, as_matrix=True, keywords=None):
#     """Computes the matrix of factors for terms in a given column.


#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         n_components: Number of components to compute.
#         as_matrix (bool): the result is reshaped by melt or not.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
#     >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
#     >>> df = pd.DataFrame(
#     ...    {
#     ...       'Authors': x,
#     ...       'Author Keywords': y,
#     ...       'Cited by': list(range(len(x))),
#     ...       'ID': list(range(len(x))),
#     ...    }
#     ... )
#     >>> df
#       Authors Author Keywords  Cited by  ID
#     0       A               a         0   0
#     1     A;B             a;b         1   1
#     2       B               b         2   2
#     3   A;B;C               c         3   3
#     4     B;D             c;d         4   4
#     5     A;B               d         5   5


#     >>> compute_tfm(df, 'Authors')
#        A  B  C  D
#     0  1  0  0  0
#     1  1  1  0  0
#     2  0  1  0  0
#     3  1  1  1  0
#     4  0  1  0  1
#     5  1  1  0  0


#     >>> factor_analysis(df, 'Authors', n_components=3)
#              F0            F1       F2
#     A -0.774597 -0.000000e+00  0.00000
#     B  0.258199  7.071068e-01 -0.57735
#     C -0.258199  7.071068e-01  0.57735
#     D  0.516398  1.110223e-16  0.57735

#     >>> factor_analysis(df, 'Authors', n_components=3, as_matrix=False)
#        Authors Factor         value
#     0        A     F0 -7.745967e-01
#     1        B     F0  2.581989e-01
#     2        C     F0 -2.581989e-01
#     3        D     F0  5.163978e-01
#     4        A     F1 -0.000000e+00
#     5        B     F1  7.071068e-01
#     6        C     F1  7.071068e-01
#     7        D     F1  1.110223e-16
#     8        A     F2  0.000000e+00
#     9        B     F2 -5.773503e-01
#     10       C     F2  5.773503e-01
#     11       D     F2  5.773503e-01

#     >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
#     >>> keywords = keywords.compile()
#     >>> factor_analysis(df, 'Authors', n_components=3, keywords=keywords)
#              F0        F1        F2
#     A -0.888074  0.000000  0.459701
#     B  0.325058  0.707107  0.627963
#     C -0.325058  0.707107 -0.627963

#     """

#     tfm = compute_tfm(x, column, keywords)
#     terms = tfm.columns.tolist()
#     if n_components is None:
#         n_components = int(np.sqrt(len(set(terms))))
#     pca = PCA(n_components=n_components)
#     result = np.transpose(pca.fit(X=tfm.values).components_)
#     result = pd.DataFrame(
#         result, columns=["F" + str(i) for i in range(n_components)], index=terms
#     )

#     if keywords is not None:
#         if keywords._patterns is None:
#             keywords = keywords.compile()
#         new_index = [w for w in result.index if w in keywords]
#         result = result.loc[new_index, :]

#     if as_matrix is True:
#         return result
#     return (
#         result.reset_index()
#         .melt("index")
#         .rename(columns={"index": column, "variable": "Factor"})
#     )


##
##
## APP
##
##

# WIDGET_WIDTH = "200px"
# LEFT_PANEL_HEIGHT = "588px"
# RIGHT_PANEL_WIDTH = "870px"
# FIGSIZE = (14, 10.0)
# PANE_HEIGHTS = ["80px", "650px", 0]

# COLUMNS = [
#     "Author Keywords",
#     "Authors",
#     "Countries",
#     "Index Keywords",
#     "Institutions",
#     "Keywords",
# ]


# def __body_0(x):
#     # -------------------------------------------------------------------------
#     #
#     # UI
#     #
#     # -------------------------------------------------------------------------
#     controls = [
#         # 0
#         {
#             "arg": "term",
#             "desc": "Term to analyze:",
#             "widget": widgets.Dropdown(
#                 options=[z for z in COLUMNS if z in x.columns],
#                 ensure_option=True,
#                 disabled=False,
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 1
#         {
#             "arg": "n_components",
#             "desc": "Number of factors:",
#             "widget": widgets.Dropdown(
#                 options=list(range(2, 21)),
#                 value=2,
#                 ensure_option=True,
#                 disabled=False,
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 2
#         {
#             "arg": "cmap",
#             "desc": "Colormap:",
#             "widget": widgets.Dropdown(
#                 options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 3
#         {
#             "arg": "filter_by",
#             "desc": "Filter by:",
#             "widget": widgets.Dropdown(
#                 options=["Frequency", "Cited by"],
#                 disable=False,
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 4
#         {
#             "arg": "filter_value",
#             "desc": "Filter value:",
#             "widget": widgets.Dropdown(
#                 options=[str(i) for i in range(10)],
#                 disable=False,
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 5
#         {
#             "arg": "sort_by",
#             "desc": "Sort order:",
#             "widget": widgets.Dropdown(
#                 options=[
#                     "Alphabetic asc.",
#                     "Alphabetic desc.",
#                     "Frequency/Cited by asc.",
#                     "Frequency/Cited by desc.",
#                 ],
#                 disable=False,
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#     ]
#     # -------------------------------------------------------------------------
#     #
#     # Logic
#     #
#     # -------------------------------------------------------------------------
#     def server(**kwargs):
#         #
#         term = kwargs["term"]
#         n_components = int(kwargs["n_components"])
#         cmap = kwargs["cmap"]
#         filter_by = kwargs["filter_by"]
#         filter_value = int(kwargs["filter_value"].split()[0])
#         sort_by = kwargs["sort_by"]
#         #
#         summ_by_term = summary_by_term(x, term)
#         #
#         a = (
#             summ_by_term[summ_by_term.columns[1]]
#             .value_counts()
#             .sort_index(ascending=False)
#         )
#         a = a.cumsum()
#         a = a.sort_index(ascending=True)
#         current_value = controls[4]["widget"].value
#         controls[4]["widget"].options = [
#             "{:d} [{:d}]".format(idx, w) for w, idx in zip(a, a.index)
#         ]
#         if current_value not in controls[4]["widget"].options:
#             controls[4]["widget"].value = controls[4]["widget"].options[0]
#         #
#         if filter_by == "Frequency":
#             filtered_terms = summ_by_term[
#                 summ_by_term["Num Documents"] >= filter_value
#             ][term]
#             num_docs = len(filtered_terms)
#         if filter_by == "Cited by":
#             filtered_terms = summ_by_term[summ_by_term["Cited by"] >= filter_value]
#             num_docs = len(filtered_terms)
#         if num_docs > 50:
#             output.clear_output()
#             with output:
#                 display(widgets.HTML("<h3>Matrix exceeds the maximum shape</h3>"))
#                 return
#         #
#         matrix = factor_analysis(
#             x=x, column=term, n_components=n_components, as_matrix=True, keywords=None
#         )
#         #
#         matrix = matrix.loc[filtered_terms, :]
#         #
#         new_names = {
#             a: "{} [{:d}]".format(a, b)
#             for a, b in zip(
#                 summ_by_term[term].tolist(), summ_by_term["Num Documents"].tolist()
#             )
#         }
#         matrix = matrix.rename(index=new_names)
#         #
#         g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
#         if sort_by == "Frequency/Cited by asc.":
#             names = sorted(matrix.index, key=g, reverse=False)
#             matrix = matrix.loc[names, :]
#         if sort_by == "Frequency/Cited by desc.":
#             names = sorted(matrix.index, key=g, reverse=True)
#             matrix = matrix.loc[names, :]
#         if sort_by == "Alphabetic asc.":
#             matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
#                 axis=1, ascending=True
#             )
#         if sort_by == "Alphabetic desc.":
#             matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
#                 axis=1, ascending=False
#             )
#         #
#         output.clear_output()
#         with output:
#             display(matrix.style.background_gradient(cmap=cmap, axis=None))

#     # -------------------------------------------------------------------------
#     #
#     # Generic
#     #
#     # -------------------------------------------------------------------------
#     args = {control["arg"]: control["widget"] for control in controls}
#     output = widgets.Output()
#     with output:
#         display(widgets.interactive_output(server, args,))
#     return widgets.HBox(
#         [
#             widgets.VBox(
#                 [
#                     widgets.VBox(
#                         [widgets.Label(value=control["desc"]), control["widget"]]
#                     )
#                     for control in controls
#                 ],
#                 layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
#             ),
#             widgets.VBox(
#                 [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
#             ),
#         ]
#     )


# def app(df):
#     """Jupyter Lab dashboard.
#     """
#     #
#     body = widgets.Tab()
#     body.children = [__body_0(df)]
#     body.set_title(0, "Matrix")
#     #
#     return AppLayout(
#         header=widgets.HTML(
#             value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
#                 "Factor Analysis"
#             )
#         ),
#         center=body,
#         pane_heights=PANE_HEIGHTS,
#     )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
