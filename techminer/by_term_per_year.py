
# """
# Analysis by Term per Year
# ==================================================================================================



# """
# import ipywidgets as widgets
# import numpy as np
# import pandas as pd
# import techminer.plots as plt
# from IPython.display import HTML, clear_output, display
# from ipywidgets import AppLayout, Layout
# from techminer.by_term import citations_by_term, documents_by_term
# from techminer.by_year import documents_by_year, citations_by_year
# from techminer.explode import __explode
# from techminer.keywords import Keywords
# from techminer.plots import COLORMAPS


# def summary_by_term_per_year(x, column, keywords=None):
#     """Computes the number of documents and citations by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited_by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited_by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> summary_by_term_per_year(df, 'Authors')
#         Authors  Year  Cited_by  Num_Documents      ID
#     0  author 0  2010        21              2  [0, 1]
#     1  author 1  2010        10              1     [0]
#     2  author 2  2010        10              1     [0]
#     3  author 1  2011        12              1     [2]
#     4  author 3  2011        13              1     [3]
#     5  author 4  2012        14              1     [4]
#     6  author 4  2014        15              1     [5]

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> summary_by_term_per_year(df, 'Authors', keywords=keywords)
#         Authors  Year  Cited_by  Num_Documents   ID
#     0  author 1  2010        10              1  [0]
#     1  author 2  2010        10              1  [0]
#     2  author 1  2011        12              1  [2]
#     3  author 3  2011        13              1  [3]

#     """
#     data = x.copy()
#     data = __explode(data[["Year", column, "Cited_by", "ID"]], column)
#     data["Num_Documents"] = 1
#     result = data.groupby([column, "Year"], as_index=False).agg(
#         {"Cited_by": np.sum, "Num_Documents": np.size}
#     )
#     result = result.assign(
#         ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
#     )
#     result["Cited_by"] = result["Cited_by"].map(lambda x: int(x))
#     if keywords is not None:
#         if keywords._patterns is None:
#             keywords = keywords.compile()
#         result = result[result[column].map(lambda w: w in keywords)]
#     result.sort_values(
#         ["Year", column], ascending=True, inplace=True, ignore_index=True,
#     )
#     #
#     groupby = result.groupby('Year').agg({'Num_Documents': np.sum, 'Cited_by':np.sum})
#     groupby = groupby.reset_index()
#     docs_by_year = {key: value for key, value in zip(groupby.Year, groupby.Num_Documents)}
#     cited_by_year = {key: value for key, value in zip(groupby.Year, groupby.Cited_by)}
#     result['Documents_by_year'] = result.Year.apply(lambda w: docs_by_year[w])
#     result['Documents_by_year'] = result.Documents_by_year.map(lambda w: 1 if w == 0 else w)
#     result['Citations_by_year'] = result.Year.apply(lambda w: cited_by_year[w])
#     result['Citations_by_year'] = result.Citations_by_year.map(lambda w: 1 if w == 0 else w)
#     result = result.assign(Perc_Num_Documents=round(result.Num_Documents / result.Documents_by_year * 100, 2))
#     result = result.assign(Perc_Cited_by=round(result.Cited_by / result.Citations_by_year * 100, 2))
#     #result.pop('Documents_by_year')
#     #result.pop('Citations_by_year')
#     #
#     return result


# def documents_by_term_per_year(x, column, as_matrix=False, keywords=None):
#     """Computes the number of documents by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         as_matrix (bool): Results are returned as a matrix.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited_by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited_by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> documents_by_term_per_year(df, 'Authors')
#         Authors  Year  Num_Documents      ID
#     0  author 0  2010              2  [0, 1]
#     1  author 1  2010              1     [0]
#     2  author 2  2010              1     [0]
#     3  author 1  2011              1     [2]
#     4  author 3  2011              1     [3]
#     5  author 4  2012              1     [4]
#     6  author 4  2014              1     [5]

#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         2         1         1         0         0
#     2011         0         1         0         1         0
#     2012         0         0         0         0         1
#     2014         0         0         0         0         1

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> documents_by_term_per_year(df, 'Authors', keywords=keywords, as_matrix=True)
#           author 1  author 2  author 3
#     2010         1         1         0
#     2011         1         0         1

#     """

#     result = summary_by_term_per_year(x, column, keywords)
#     result.pop("Cited_by")
#     result.pop("Perc_Num_Documents")
#     result.pop("Perc_Cited_by")
#     result.sort_values(
#         ["Year", "Num_Documents", column], ascending=[True, False, True], inplace=True,
#     )
#     result.reset_index(drop=True)
#     if as_matrix == True:
#         result = pd.pivot_table(
#             result, values="Num_Documents", index="Year", columns=column, fill_value=0,
#         )
#         result.columns = result.columns.tolist()
#         result.index = result.index.tolist()
#     return result

# def perc_documents_by_term_per_year(x, column, as_matrix=False, keywords=None):
#     """Computes the number of documents by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         as_matrix (bool): Results are returned as a matrix.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited_by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited_by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> documents_by_term_per_year(df, 'Authors')
#         Authors  Year  Num_Documents      ID
#     0  author 0  2010              2  [0, 1]
#     1  author 1  2010              1     [0]
#     2  author 2  2010              1     [0]
#     3  author 1  2011              1     [2]
#     4  author 3  2011              1     [3]
#     5  author 4  2012              1     [4]
#     6  author 4  2014              1     [5]

#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         2         1         1         0         0
#     2011         0         1         0         1         0
#     2012         0         0         0         0         1
#     2014         0         0         0         0         1

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> documents_by_term_per_year(df, 'Authors', keywords=keywords, as_matrix=True)
#           author 1  author 2  author 3
#     2010         1         1         0
#     2011         1         0         1

#     """

#     result = summary_by_term_per_year(x, column, keywords)
#     result.pop("Cited_by")
#     result.pop("Num_Documents")
#     result.pop('Perc_Cited_by')
#     result.sort_values(
#         ["Year", "Perc_Num_Documents", column], ascending=[True, False, True], inplace=True,
#     )
#     result.reset_index(drop=True)
#     if as_matrix == True:
#         result = pd.pivot_table(
#             result, values="Perc_Num_Documents", index="Year", columns=column, fill_value=0,
#         )
#         result.columns = result.columns.tolist()
#         result.index = result.index.tolist()
#     return result


# def gant(x, column, keywords=None):
#     """Computes the number of documents by term per year.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         as_matrix (bool): Results are returned as a matrix.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2011, 2011, 2012, 2015, 2012, 2016],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 3,author 4,author 4".split(","),
#     ...          "Cited_by": list(range(10,17)),
#     ...          "ID": list(range(7)),
#     ...     }
#     ... )
#     >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         1         1         0         0
#     2011         1         1         0         0         0
#     2012         0         0         0         1         1
#     2015         0         0         0         1         0
#     2016         0         0         0         0         1

#     >>> gant(df, 'Authors')
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         1         1         0         0
#     2011         1         1         0         0         0
#     2012         0         0         0         1         1
#     2013         0         0         0         1         1
#     2014         0         0         0         1         1
#     2015         0         0         0         1         1
#     2016         0         0         0         0         1

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> gant(df, 'Authors', keywords=keywords)
#           author 1  author 2  author 3
#     2010         1         1         0
#     2011         1         0         0
#     2012         0         0         1
#     2013         0         0         1
#     2014         0         0         1
#     2015         0         0         1

#     """
#     result = documents_by_term_per_year(
#         x, column=column, as_matrix=True, keywords=keywords
#     )
#     years = [year for year in range(result.index.min(), result.index.max() + 1)]
#     result = result.reindex(years, fill_value=0)
#     matrix1 = result.copy()
#     matrix1 = matrix1.cumsum()
#     matrix1 = matrix1.applymap(lambda x: True if x > 0 else False)
#     matrix2 = result.copy()
#     matrix2 = matrix2.sort_index(ascending=False)
#     matrix2 = matrix2.cumsum()
#     matrix2 = matrix2.applymap(lambda x: True if x > 0 else False)
#     matrix2 = matrix2.sort_index(ascending=True)
#     result = matrix1.eq(matrix2)
#     result = result.applymap(lambda x: 1 if x is True else 0)
#     return result


# def citations_by_term_per_year(x, column, as_matrix=False, keywords=None):
#     """Computes the number of citations by term by year in a column.

#     Args:
#         column (str): the column to explode.
#         as_matrix (bool): Results are returned as a matrix.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited_by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited_by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> citations_by_term_per_year(df, 'Authors')
#         Authors  Year  Cited_by      ID
#     0  author 0  2010        21  [0, 1]
#     1  author 2  2010        10     [0]
#     2  author 1  2010        10     [0]
#     3  author 3  2011        13     [3]
#     4  author 1  2011        12     [2]
#     5  author 4  2012        14     [4]
#     6  author 4  2014        15     [5]

#     >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010        21        10        10         0         0
#     2011         0        12         0        13         0
#     2012         0         0         0         0        14
#     2014         0         0         0         0        15

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> citations_by_term_per_year(df, 'Authors', keywords=keywords)
#         Authors  Year  Cited_by   ID
#     0  author 2  2010        10  [0]
#     1  author 1  2010        10  [0]
#     2  author 3  2011        13  [3]
#     3  author 1  2011        12  [2]

#     """
#     result = summary_by_term_per_year(x, column, keywords)
#     result.pop("Num_Documents")
#     result.sort_values(
#         ["Year", "Cited_by", column], ascending=[True, False, False], inplace=True,
#     )
#     result = result.reset_index(drop=True)
#     if as_matrix == True:
#         result = pd.pivot_table(
#             result, values="Cited_by", index="Year", columns=column, fill_value=0,
#         )
#         result.columns = result.columns.tolist()
#         result.index = result.index.tolist()
#     return result

# def perc_citations_by_term_per_year(x, column, as_matrix=False, keywords=None):
#     """Computes the number of citations by term by year in a column.

#     Args:
#         column (str): the column to explode.
#         as_matrix (bool): Results are returned as a matrix.
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#     ...          "Cited_by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#        Year                     Authors  Cited_by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2014                    author 4        15   5

#     >>> citations_by_term_per_year(df, 'Authors')
#         Authors  Year  Cited_by      ID
#     0  author 0  2010        21  [0, 1]
#     1  author 2  2010        10     [0]
#     2  author 1  2010        10     [0]
#     3  author 3  2011        13     [3]
#     4  author 1  2011        12     [2]
#     5  author 4  2012        14     [4]
#     6  author 4  2014        15     [5]

#     >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010        21        10        10         0         0
#     2011         0        12         0        13         0
#     2012         0         0         0         0        14
#     2014         0         0         0         0        15

#     >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
#     >>> keywords = keywords.compile()
#     >>> citations_by_term_per_year(df, 'Authors', keywords=keywords)
#         Authors  Year  Cited_by   ID
#     0  author 2  2010        10  [0]
#     1  author 1  2010        10  [0]
#     2  author 3  2011        13  [3]
#     3  author 1  2011        12  [2]

#     """
#     result = summary_by_term_per_year(x, column, keywords)
#     result.pop("Num_Documents")
#     result.pop("Cited_by")
#     result.pop("Perc_Num_Documents")
#     result.sort_values(
#         ["Year", "Perc_Cited_by", column], ascending=[True, False, False], inplace=True,
#     )
#     result = result.reset_index(drop=True)
#     if as_matrix == True:
#         result = pd.pivot_table(
#             result, values="Perc_Cited_by", index="Year", columns=column, fill_value=0,
#         )
#         result.columns = result.columns.tolist()
#         result.index = result.index.tolist()
#     return result


# def growth_indicators(x, column, timewindow=2, keywords=None):
#     """Computes the average growth rate of a group of terms.

#     Args:
#         column (str): the column to explode.
#         timewindow (int): time window for analysis
#         keywords (Keywords): filter the result using the specified Keywords object.

#     Returns:
#         DataFrame.


#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...   {
#     ...     "Year": [2010, 2010, 2011, 2011, 2012, 2013, 2014, 2014],
#     ...     "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4,author 0;author 3,author 3;author 4".split(","),
#     ...     "Cited_by": list(range(10,18)),
#     ...     "ID": list(range(8)),
#     ...   }
#     ... )
#     >>> x
#        Year                     Authors  Cited_by  ID
#     0  2010  author 0;author 1;author 2        10   0
#     1  2010                    author 0        11   1
#     2  2011                    author 1        12   2
#     3  2011                    author 3        13   3
#     4  2012                    author 4        14   4
#     5  2013                    author 4        15   5
#     6  2014           author 0;author 3        16   6
#     7  2014           author 3;author 4        17   7

#     >>> documents_by_term_per_year(x, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         2         1         1         0         0
#     2011         0         1         0         1         0
#     2012         0         0         0         0         1
#     2013         0         0         0         0         1
#     2014         1         0         0         2         1

#     >>> growth_indicators(x, 'Authors')
#         Authors       AGR  ADY   PDLY  Before 2013  Between 2013-2014
#     0  author 3  0.666667  1.0  12.50            1                  2
#     1  author 0  0.333333  0.5   6.25            2                  1
#     2  author 4  0.000000  1.0  12.50            1                  2

#     >>> keywords = Keywords(['author 3', 'author 4'])
#     >>> keywords = keywords.compile()
#     >>> growth_indicators(x, 'Authors', keywords=keywords)
#         Authors       AGR  ADY  PDLY  Before 2013  Between 2013-2014
#     0  author 3  0.666667  1.0  12.5            1                  2
#     1  author 4  0.000000  1.0  12.5            1                  2

#     """

#     def compute_agr():
#         result = documents_by_term_per_year(
#             x, column=column, keywords=keywords
#         )
#         years_agr = sorted(set(result.Year))[-(timewindow + 1) :]
#         years_agr = [years_agr[0], years_agr[-1]]
#         result = result[result.Year.map(lambda w: w in years_agr)]
#         result.pop("ID")
#         result = pd.pivot_table(
#             result,
#             columns="Year",
#             index=column,
#             values="Num_Documents",
#             fill_value=0,
#         )
#         result["AGR"] = 0.0
#         result = result.assign(
#             AGR=(result[years_agr[1]] - result[years_agr[0]]) / (timewindow + 1)
#         )
#         result.pop(years_agr[0])
#         result.pop(years_agr[1])
#         result.columns = list(result.columns)
#         result = result.sort_values(by=["AGR", column], ascending=False)
#         result.reset_index(drop=True)
#         return result

#     def compute_ady():
#         result = documents_by_term_per_year(
#             x, column=column, keywords=keywords
#         )
#         years_ady = sorted(set(result.Year))[-timewindow:]
#         result = result[result.Year.map(lambda w: w in years_ady)]
#         result = result.groupby([column], as_index=False).agg(
#             {"Num_Documents": np.sum}
#         )
#         result = result.rename(columns={"Num_Documents": "ADY"})
#         result["ADY"] = result.ADY.map(lambda w: w / timewindow)
#         result = result.reset_index(drop=True)
#         return result

#     def compute_num_documents():
#         result = documents_by_term_per_year(
#             x, column=column, keywords=keywords
#         )
#         years_between = sorted(set(result.Year))[-timewindow:]
#         years_before = sorted(set(result.Year))[0:-timewindow]
#         between = result[result.Year.map(lambda w: w in years_between)]
#         before = result[result.Year.map(lambda w: w in years_before)]
#         between = between.groupby([column], as_index=False).agg(
#             {"Num_Documents": np.sum}
#         )
#         between = between.rename(
#             columns={
#                 "Num_Documents": "Between {}-{}".format(
#                     years_between[0], years_between[-1]
#                 )
#             }
#         )
#         before = before.groupby([column], as_index=False).agg(
#             {"Num_Documents": np.sum}
#         )
#         before = before.rename(
#             columns={"Num_Documents": "Before {}".format(years_between[0])}
#         )
#         result = pd.merge(before, between, on=column)
#         # result = result.set_index(column)
#         return result

#     result = compute_agr()
#     # result = result.set_index(column)
#     ady = compute_ady()
#     result = pd.merge(result, ady, on=column)
#     result = result.assign(PDLY=round(result.ADY / len(x) * 100, 2))
#     num_docs = compute_num_documents()
#     result = pd.merge(result, num_docs, on=column)
#     result = result.reset_index(drop=True)
#     return result



# ##
# ##
# ## APP
# ##
# ##

# WIDGET_WIDTH = "200px"
# LEFT_PANEL_HEIGHT = "588px"
# RIGHT_PANEL_WIDTH = "870px"
# FIGSIZE = (14, 10.0)
# PANE_HEIGHTS = ["80px", "650px", 0]






# ##
# ##
# ##  Time Analysis
# ##
# ##
# def __body_0(x):
#     COLUMNS = [
#         "Author_Keywords",
#         "Author_Keywords_CL",
#         "Authors",
#         "Countries",
#         "Country_1st_Author",
#         "Document_type",
#         "Index_Keywords",
#         "Index_Keywords_CL",
#         "Institution_1st_Author",
#         "Institutions",
#         "Source title",
#     ]
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
#             "widget": widgets.Select(
#                 options=[z for z in COLUMNS if z in x.columns],
#                 ensure_option=True,
#                 disabled=False,
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 1
#         {
#             "arg": "analysis_type",
#             "desc": "Analysis type:",
#             "widget": widgets.Dropdown(
#                 options=["Frequency", "Citation", "Perc. Frequency", "Perc. Cited by"],
#                 value="Frequency",
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 2
#         {
#             "arg": "plot_type",
#             "desc": "View:",
#             "widget": widgets.Dropdown(
#                 options=["Heatmap", "Bubble plot", "Gant diagram", 'Lines plot', "Table"],
#                 layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 3
#         {
#             "arg": "cmap",
#             "desc": "Colormap:",
#             "widget": widgets.Dropdown(
#                 options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 4
#         {
#             "arg": "top_n",
#             "desc": "Top N:",
#             "widget": widgets.IntSlider(
#                 value=10,
#                 min=5,
#                 max=60,
#                 step=1,
#                 continuous_update=False,
#                 orientation="horizontal",
#                 readout=True,
#                 readout_format="d",
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
#         analysis_type = kwargs["analysis_type"]
#         plot_type = kwargs["plot_type"]
#         cmap = kwargs["cmap"]
#         top_n = kwargs["top_n"]
#         #
#         plots = {"Heatmap": plt.heatmap, "Gant diagram": plt.gant, "Bubble plot": plt.bubble, "Lines": plt.plot, "Table":None}
#         plot = plots[plot_type]
#         #
#         if analysis_type == "Frequency":
#             top = documents_by_term(x, term).head(top_n)[term].tolist()
#             matrix = documents_by_term_per_year(x, term, as_matrix=True)
#         if analysis_type == 'Citation':
#             top = citations_by_term(x, term).head(top_n)[term].tolist()
#             matrix = citations_by_term_per_year(x, term, as_matrix=True)
#         if analysis_type == 'Perc. Frequency':
#             top = citations_by_term(x, term).head(top_n)[term].tolist()
#             matrix = perc_documents_by_term_per_year(x, term, as_matrix=True)
#         if analysis_type == 'Perc. Cited by':
#             top = citations_by_term(x, term).head(top_n)[term].tolist()
#             matrix = perc_citations_by_term_per_year(x, term, as_matrix=True)
#         matrix = matrix[top]
#         output.clear_output()
#         with output:
#             if plot_type == "Heatmap":
#                 display(plot(matrix, cmap=cmap, figsize=FIGSIZE))
#             if plot_type == "Gant diagram":
#                 display(plot(matrix, figsize=FIGSIZE))
#             if plot_type == "Bubble plot":
#                 display(plot(matrix.transpose(), axis=0, cmap=cmap, figsize=FIGSIZE))
#             if plot_type == "Lines plot":
#                 display(plot(matrix, cmap=cmap, figsize=FIGSIZE))
#             if plot_type == 'Table':
#                 with pd.option_context("display.max_columns", 60):
#                     display(matrix)


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
#             widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
#         ]
#     )


# ##
# ##
# ##  Growth Indicators
# ##
# ##
# def __body_1(x):
#     #
#     COLUMNS = [
#         "Author_Keywords",
#         "Author_Keywords_CL",
#         "Countries",
#         "Country_1st_Author",
#         "Index_Keywords",
#         "Index_Keywords_CL",
#         "Institution_1st_Author",
#         "Institutions",
#     ]
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
#             "widget": widgets.Select(
#                     options=[z for z in COLUMNS if z in x.columns],
#                     ensure_option=True,
#                     layout=Layout(width=WIDGET_WIDTH),
#                 ),
#         },
#         # 1
#         {
#             "arg": "analysis_type",
#             "desc": "Analysis type:",
#             "widget": widgets.Dropdown(
#                     options=[
#                         "Average Growth Rate",
#                         "Average Documents per Year",
#                         "Percentage of Documents in Last Years",
#                         "Number of Document Published",
#                     ],
#                     value="Average Growth Rate",
#                     layout=Layout(width=WIDGET_WIDTH),
#                 ),
#         },
#         # 2
#         {
#             "arg": "time_window",
#             "desc": "Time window:",
#             "widget": widgets.Dropdown(
#                     options=["2", "3", "4", "5"],
#                     value="2",
#                     layout=Layout(width=WIDGET_WIDTH),
#                 ),
#         },
#         # 3
#         {
#             "arg": "plot_type",
#             "desc": "Plot type:",
#             "widget": widgets.Dropdown(
#                     options=["bar", "barh"],
#                     layout=Layout(width=WIDGET_WIDTH),
#                 ),
#         },
#         # 4
#         {
#             "arg": "cmap",
#             "desc": "Colormap:",
#             "widget": widgets.Dropdown(
#                 options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
#             ),
#         },
#         # 5
#         {
#             "arg": "top_n",
#             "desc": "Top N:",
#             "widget": widgets.IntSlider(
#                     value=10,
#                     min=10,
#                     max=100,
#                     step=1,
#                     continuous_update=False,
#                     orientation="horizontal",
#                     readout=True,
#                     readout_format="d",
#                     layout=Layout(width=WIDGET_WIDTH),
#                 ),
#         },
#     ]
#     # -------------------------------------------------------------------------
#     #
#     # Logic
#     #
#     # -------------------------------------------------------------------------
#     def server(**kwargs):
#         #
#         term = kwargs['term']
#         cmap = kwargs['cmap']
#         analysis_type = kwargs['analysis_type']
#         top_n = kwargs['top_n']
#         plot_type = kwargs['plot_type']
#         time_window = int(kwargs['time_window'])
#         #
#         plots = {"bar": plt.bar, "barh": plt.barh}
#         plot = plots[plot_type]
#         #
#         df = growth_indicators(x, term, timewindow=time_window)
#         output.clear_output()
#         if plot_type == 'bar':
#             figsize = (14, 7.5)
#         if plot_type == 'barh':
#             figsize = (14, 10.0)

#         with output:
#             if analysis_type == "Average Growth Rate":
#                 df = df.sort_values('AGR', ascending=False).head(top_n)
#                 df = df.reset_index(drop=True)
#                 display(plot(df[[term, 'AGR']], cmap=cmap, figsize=figsize))
#             if analysis_type == "Average Documents per Year":
#                 df = df.sort_values('ADY', ascending=False).head(top_n)
#                 df = df.reset_index(drop=True)
#                 display(plot(df[[term, 'ADY']], cmap=cmap, figsize=figsize))
#             if analysis_type == "Percentage of Documents in Last Years":
#                 df = df.sort_values('PDLY', ascending=False).head(top_n)
#                 df = df.reset_index(drop=True)
#                 display(plot(df[[term, 'PDLY']], cmap=cmap, figsize=figsize))
#             if analysis_type == "Number of Document Published":
#                 df['Num_Documents'] = df[df.columns[-2]] + df[df.columns[-1]]
#                 df = df.sort_values('Num_Documents', ascending=False).head(top_n)
#                 df = df.reset_index(drop=True)
#                 df.pop('Num_Documents')
#                 if plot_type == 'bar':
#                     display(plt.stacked_bar(df[[term, df.columns[-2], df.columns[-1]]], figsize=figsize, cmap=cmap))
#                 if plot_type == 'barh':
#                     display(plt.stacked_barh(df[[term, df.columns[-2], df.columns[-1]]], figsize=figsize, cmap=cmap))


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
#             widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
#         ]
#     )



# def app(df):
#     """Jupyter Lab dashboard.
#     """
#     #
#     body = widgets.Tab()
#     body.children = [__body_0(df), __body_1(df)]
#     body.set_title(0, "Time Analysis")
#     body.set_title(1, "Growth Indicators")
#     #
#     return AppLayout(
#         header=widgets.HTML(
#             value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
#                 "Summary by Term per Year"
#             )
#         ),
#         center=body,
#         pane_heights=PANE_HEIGHTS,
#     )



# #
# #
# #
# if __name__ == "__main__":

#     import doctest

#     doctest.testmod()
