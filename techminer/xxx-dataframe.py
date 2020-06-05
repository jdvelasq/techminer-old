"""
DataFrame object
==================================================================================================

Column names
--------------------------------------------------------------------------------------------------

The column names in the dataframe follows the convetion used in WoS.

* `AB`: Abstract.
* `AF`: Author full name.
* `AR`: Article Number.
* `AU`: Authors.
* `AI`: Authors Identifiers.
* `BA`: Book Authors.
* `BE`: Editors.
* `BF`: Book Authors Full Name.
* `BN`: International Standard Book Number (ISBN).
* `BP`: Begining page.
* `BS`: Book Series Subtitle.
* `C1`: Author Address.
* `CA`: Group Authors.
* `CL`: Conference Location.
* `CR`: Cited References.
* `CT`: Conference Title.
* `CY`: Conference Date.
* `D2`: Book DOI.
* `DA`:	Date this report was generated.
* `DE`: Author keywords.
* `DI`: DOI
* `DT`: Document Type.
* `EA`: Early access date.
* `EF`:	End of File.
* `EI`: Electronic International Standard Serial Number (eISSN).
* `EM`: E-mail Address.
* `EP`: Ending page.
* `ER`:	End of Record.
* `EY`: Early access year.
* `FN`: File Name.
* `FU`: Funding Agency and Grant Number.
* `FX`: Funding Text.
* `GA`:	Document Delivery Number.
* `GP`: Book Group Authors.
* `HC`:	ESI Highly Cited Paper. Note that this field is valued only for ESI subscribers.
* `HO`: Conference Host.
* `HP`:	ESI Hot Paper. Note that this field is valued only for ESI subscribers.
* `ID`: Keyword plus.
* `IS`: Issue.
* `J9`: 29-Character Source Abbreviation.
* `JI`: ISO Source Abbreviation
* `LA`: Language.
* `MA`: Meeting Abstract.
* `NR`: Cited Reference Count.
* `OA`:	Open Access Indicator.
* `OI`: ORCID Identifier (Open Researcher and Contributor ID).
* `P2`: Chapter Count (Book Citation Index).
* `PA`: Publisher Address.
* `PG`: Page count.
* `PI`: Publisher City.
* `PM`:	PubMed ID.
* `PN`: Part Number.
* `PR`: Reprint Address.
* `PT`: Publication Type (J=Journal; B=Book; S=Series; P=Patent).
* `PU`: Publisher.
* `PY`: Year Published.
* `RI`: ResearcherID Number.
* `SC`: Research Areas.
* `SE`: Book Series Title.
* `SI`: Special Issue.
* `SN`: International Standard Serial Number (ISSN).
* `SO`: Publication Name.
* `SP`: Conference Sponsors.
* `SU`: Supplement.
* `TC`: Web of Science Core Collection Times Cited Count.
* `TI`: Document Title.
* `U1`: Usage Count (Last 180 Days).
* `U2`: Usage Count (Since 2013).
* `UT`:	Accession Number.
* `VL`: Volume.
* `VR`: Version Number.
* `WC`: Web of Science Categories.
* `Z9`: Total Times Cited Count.


"""
# import json
# import math
# import re
# from os.path import dirname, join

# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA

# from .keywords import Keywords
#  from .text import remove_accents

# SCOPUS_SEPS = {
#     "Authors": ",",
#     "Author(s) ID": ";",
#     "Author Keywords": ";",
#     "Index Keywords": ";",
#     "ID": ";",
#     "Keywords": ";",
# }

# SCOPUS_COLS = [
#     "Authors",
#     "Author(s) ID",
#     "Author Keywords",
#     "Index Keywords",
#     "ID",
#     "Keywords",
# ]


# def relationship(x, y):
#     sxy = sum([a * b * min(a, b) for a, b in zip(x, y)])
#     a = math.sqrt(sum(x))
#     b = math.sqrt(sum(y))
#     return sxy / (a * b)


# def heatmap(df, cmap="Blues"):
#     """Display the dataframe as a heatmap in Jupyter.

#         Args:
#             df (pandas.DataFrame): dataframe as a matrix.
#             cmap (colormap): colormap used to build a heatmap.

#         Returns:
#             DataFrame. Exploded dataframe.


#     """
#     return df.style.background_gradient(cmap=cmap)


# def sort_by_numdocuments(
#     df, matrix, axis=0, ascending=True, kind="quicksort", axis_name=None, axis_sep=None
# ):
#     """Sorts a matrix axis by the number of documents.


#     Args:
#         df (pandas.DataFrame): dataframe with bibliographic information.
#         matrix (pandas.DataFrame): matrix to sort.
#         axis ({0 or ‘index’, 1 or ‘columns’}), default 0: axis to be sorted.
#         ascending (bool): sort ascending?.
#         kind (str): ‘quicksort’, ‘mergesort’, ‘heapsort’.
#         axis_name (str): column name used to sort by number of documents.
#         axis_sep (str): character used to separate the internal values of column axis_name.

#     Returns:
#         DataFrame sorted.

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...         "c0": ["D"] * 4 + ["B"] * 3 + ["C"] * 2 + ["A"],
#     ...         "c1": ["a"] * 4 + ["c"] * 3 + ["b"] * 2 + ["d"],
#     ...         "Cited by": list(range(10)),
#     ...         "ID": list(range(10)),
#     ...     },
#     ... )
#     >>> df
#       c0 c1  Cited by  ID
#     0  D  a         0   0
#     1  D  a         1   1
#     2  D  a         2   2
#     3  D  a         3   3
#     4  B  c         4   4
#     5  B  c         5   5
#     6  B  c         6   6
#     7  C  b         7   7
#     8  C  b         8   8
#     9  A  d         9   9

#     >>> matrix = pd.DataFrame(
#     ...     {"D": [0, 1, 2, 3], "B": [4, 5, 6, 7], "A": [8, 9, 10, 11], "C": [12, 13, 14, 15],},
#     ...     index=list("badc"),
#     ... )
#     >>> matrix
#        D  B   A   C
#     b  0  4   8  12
#     a  1  5   9  13
#     d  2  6  10  14
#     c  3  7  11  15

#     >>> sort_by_numdocuments(df, matrix, axis='columns', ascending=True, axis_name='c0')
#         A  B   C  D
#     b   8  4  12  0
#     a   9  5  13  1
#     d  10  6  14  2
#     c  11  7  15  3

#     >>> sort_by_numdocuments(df, matrix, axis='columns', ascending=False, axis_name='c0')
#        D   C  B   A
#     b  0  12  4   8
#     a  1  13  5   9
#     d  2  14  6  10
#     c  3  15  7  11

#     >>> sort_by_numdocuments(df, matrix, axis='index', ascending=True, axis_name='c1')
#        D  B   A   C
#     a  1  5   9  13
#     b  0  4   8  12
#     c  3  7  11  15
#     d  2  6  10  14

#     >>> sort_by_numdocuments(df, matrix, axis='index', ascending=False, axis_name='c1')
#        D  B   A   C
#     d  2  6  10  14
#     c  3  7  11  15
#     b  0  4   8  12
#     a  1  5   9  13

#     """
#     terms = DataFrame(df).documents_by_term(column=axis_name, sep=axis_sep)
#     terms_sorted = (
#         terms.sort_values(by=axis_name, kind=kind, ascending=ascending)
#         .iloc[:, 0]
#         .tolist()
#     )
#     if axis == "index":
#         return matrix.loc[terms_sorted, :]
#     return matrix.loc[:, terms_sorted]


# def sort_by_citations(
#     df, matrix, axis=0, ascending=True, kind="quicksort", axis_name=None, axis_sep=None
# ):
#     """Sorts a matrix axis by the citations.


#     Args:
#         df (pandas.DataFrame): dataframe with bibliographic information.
#         matrix (pandas.DataFrame): matrix to sort.
#         axis ({0 or ‘index’, 1 or ‘columns’}), default 0: axis to be sorted.
#         ascending (bool): sort ascending?.
#         kind (str): ‘quicksort’, ‘mergesort’, ‘heapsort’.
#         axis_name (str): column name used to sort by citations.
#         axis_sep (str): character used to separate the internal values of column axis_name.

#     Returns:
#         DataFrame sorted.

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...         "c0": ["D"] * 4 + ["B"] * 3 + ["C"] * 2 + ["A"],
#     ...         "c1": ["a"] * 4 + ["c"] * 3 + ["b"] * 2 + ["d"],
#     ...         "Cited by": list(range(10)),
#     ...         "ID": list(range(10)),
#     ...     },
#     ... )
#     >>> df
#       c0 c1  Cited by  ID
#     0  D  a         0   0
#     1  D  a         1   1
#     2  D  a         2   2
#     3  D  a         3   3
#     4  B  c         4   4
#     5  B  c         5   5
#     6  B  c         6   6
#     7  C  b         7   7
#     8  C  b         8   8
#     9  A  d         9   9

#     >>> matrix = pd.DataFrame(
#     ...     {"D": [0, 1, 2, 3], "B": [4, 5, 6, 7], "A": [8, 9, 10, 11], "C": [12, 13, 14, 15],},
#     ...     index=list("badc"),
#     ... )
#     >>> matrix
#        D  B   A   C
#     b  0  4   8  12
#     a  1  5   9  13
#     d  2  6  10  14
#     c  3  7  11  15

#     >>> sort_by_citations(df, matrix, axis='columns', ascending=True, axis_name='c0')
#         A  B   C  D
#     b   8  4  12  0
#     a   9  5  13  1
#     d  10  6  14  2
#     c  11  7  15  3

#     >>> sort_by_citations(df, matrix, axis='columns', ascending=False, axis_name='c0')
#        D   C  B   A
#     b  0  12  4   8
#     a  1  13  5   9
#     d  2  14  6  10
#     c  3  15  7  11

#     >>> sort_by_citations(df, matrix, axis='index', ascending=True, axis_name='c1')
#        D  B   A   C
#     a  1  5   9  13
#     b  0  4   8  12
#     c  3  7  11  15
#     d  2  6  10  14

#     >>> sort_by_citations(df, matrix, axis='index', ascending=False, axis_name='c1')
#        D  B   A   C
#     d  2  6  10  14
#     c  3  7  11  15
#     b  0  4   8  12
#     a  1  5   9  13

#     """
#     terms = DataFrame(df).citations_by_term(column=axis_name, sep=axis_sep)
#     terms_sorted = (
#         terms.sort_values(by=axis_name, kind=kind, ascending=ascending)
#         .iloc[:, 0]
#         .tolist()
#     )
#     if axis == "index":
#         return matrix.loc[terms_sorted, :]
#     return matrix.loc[:, terms_sorted]


# class DataFrame(pd.DataFrame):
#     """Data structure derived from a `pandas:DataFrame
#     <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame>`_.
#     object with specialized functions to summarize and query bibliographic data.


#     """

#     ##
#     ##
#     ##  Compatitbility with pandas.DataFrame
#     ##
#     ##

#     @property
#     def _constructor_expanddim(self):
#         return self


#     ##
#     ##
#     ##  Data preparation functions
#     ##
#     ##

#     def keywords_fusion(
#         self,
#         column="Author Keywords",
#         sep=None,
#         other="Index Keywords",
#         sep_other=None,
#         new_column="Keywords",
#         sep_new_column=None,
#     ):
#         """Combines keywords columns.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...   {
#         ...      "Author Keywords": "k0;k2;,k1;,k3;,k4;,k5;".split(","),
#         ...      "Index Keywords": "k0;k1;,k2;k3;,k3;,k4;,k6;".split(','),
#         ...   }
#         ... )
#         >>> df
#           Author Keywords Index Keywords
#         0          k0;k2;         k0;k1;
#         1             k1;         k2;k3;
#         2             k3;            k3;
#         3             k4;            k4;
#         4             k5;            k6;
#         >>> DataFrame(df).keywords_fusion()
#           Author Keywords Index Keywords  Keywords
#         0          k0;k2;         k0;k1;  k0;k1;k2
#         1             k1;         k2;k3;  k1;k2;k3
#         2             k3;            k3;        k3
#         3             k4;            k4;        k4
#         4             k5;            k6;     k5;k6

#         """

#         sep = ";" if sep is None and column in SCOPUS_COLS else sep
#         sep_other = ";" if sep_other is None and column in SCOPUS_COLS else sep_other
#         sep_new_column = (
#             ";"
#             if sep_new_column is None and new_column in SCOPUS_COLS
#             else sep_new_column
#         )

#         df = self.copy()
#         author_keywords = df[column].map(
#             lambda x: x.split(sep) if x is not None else []
#         )
#         index_keywords = df[other].map(
#             lambda x: x.split(sep_other) if x is not None else []
#         )
#         keywords = author_keywords + index_keywords
#         keywords = keywords.map(lambda x: [e for e in x if e != ""])
#         keywords = keywords.map(lambda x: [e.strip() for e in x])
#         keywords = keywords.map(lambda x: sorted(set(x)))
#         keywords = keywords.map(lambda x: sep_new_column.join(x))
#         keywords = keywords.map(lambda x: None if x == "" else x)
#         df[new_column] = keywords
#         return df

#     def keywords_completation(self):
#         """Complete keywords in 'Keywords' column (if exists) from title and abstract.
#         """

#         cp = self.copy()
#         if "Keywords" not in self.columns:
#             cp = cp.keywords_fusion()

#         # Remove copyright character from abstract
#         abstract = cp.Abstract.map(
#             lambda x: x[0 : x.find("\u00a9")]
#             if isinstance(x, str) and x.find("\u00a9") != -1
#             else x
#         )

#         title_abstract = cp.Title + " " + abstract

#         kyw = Keywords()
#         kyw.add_keywords(cp.Keywords, sep=";")

#         keywords = title_abstract.map(lambda x: kyw.extract_from_text(x, sep=";"))
#         idx = cp.Keywords.map(lambda x: x is None)
#         cp.loc[idx, "Keywords"] = keywords[idx]

#         return DataFrame(cp)

#     def generate_ID(self, fmt=None):
#         """Generates a sequence of integers as ID for each document.


#         Args:
#             fmt (str): Format used to generate ID column.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...   {
#         ...     "Authors": "author 0,author 1,author 2;author 3;author 4".split(";"),
#         ...   }
#         ... )
#         >>> df
#                               Authors
#         0  author 0,author 1,author 2
#         1                    author 3
#         2                    author 4

#         >>> DataFrame(df).generate_ID()
#                               Authors  ID
#         0  author 0,author 1,author 2   0
#         1                    author 3   1
#         2                    author 4   2

#         """
#         if fmt is None:
#             self["ID"] = [x for x in range(len(self))]
#         else:
#             self["ID"] = [fmt.format(x) for x in range(len(self))]
#         return DataFrame(self)


#     # def remove_accents(self):
#     #     """Remove accents for all strings on a DataFrame

#     #     Returns:
#     #         A dataframe.

#     #     Examples
#     #     ----------------------------------------------------------------------------------------------

#     #     >>> import pandas as pd
#     #     >>> df = pd.DataFrame(
#     #     ...   {
#     #     ...      "Authors": "áàÁÀ;éèÉÈ;íìÌÍ;ÑÑÑñññ".split(";"),
#     #     ...   }
#     #     ... )
#     #     >>> DataFrame(df).remove_accents()
#     #       Authors
#     #     0    aaAA
#     #     1    eeEE
#     #     2    iiII
#     #     3  NNNnnn

#     #     """
#     #     return DataFrame(
#     #         self.applymap(lambda x: remove_accents(x) if isinstance(x, str) else x)
#     #     )

#     ##
#     ##
#     ##  Row selection
#     ##
#     ##

#     def get_rows_by_IDs(self, IDs):
#         """Extracts records using the ID number.
#         """
#         IDs = [e for x in IDs for e in x]
#         result = self[self["ID"].map(lambda x: x in IDs)]
#         return DataFrame(result)

#     ##
#     ##
#     ## Basic data information
#     ##
#     ##


#     def most_cited_documents(self):
#         """ Returns the cited documents.

#         Results:
#             pandas.DataFrame

#         """
#         result = self.sort_values(by="Cited by", ascending=False)[
#             ["Title", "Authors", "Year", "Cited by", "ID"]
#         ]
#         result["Cited by"] = result["Cited by"].map(
#             lambda x: int(x) if pd.isna(x) is False else 0
#         )
#         return result

#     def most_cited_authors(self):
#         """ Returns the cited authors.

#         Results:
#             pandas.DataFrame

#         """
#         result = self.summarize_by_term("Authors")
#         result = result.sort_values(by="Cited by", ascending=False)
#         return result


#     def summarize_by_term_per_term_per_year(
#         self, column_IDX, column_COL, sep_IDX=None, sep_COL=None, keywords=None
#     ):
#         """Computes the number of documents and citations by term per term by year.

#         Args:
#             column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
#             sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
#             column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
#             sep_COL (str): Character used as internal separator for the elements in the column_COL.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#         ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
#         ...          "Cited by": list(range(10,16)),
#         ...          "ID": list(range(6)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors Author Keywords  Cited by  ID
#         0  2010  author 0;author 1;author 2           w0;w1        10   0
#         1  2010                    author 0              w0        11   1
#         2  2011                    author 1              w1        12   2
#         3  2011                    author 3        w5;w3;w4        13   3
#         4  2012                    author 4              w5        14   4
#         5  2014                    author 4              w3        15   5

#         >>> DataFrame(df).summarize_by_term_per_term_per_year('Authors', 'Author Keywords')
#              Authors Author Keywords  Year  Cited by  Num Documents      ID
#         0   author 0              w0  2010        21              2  [0, 1]
#         1   author 0              w1  2010        10              1     [0]
#         2   author 1              w0  2010        10              1     [0]
#         3   author 1              w1  2010        10              1     [0]
#         4   author 2              w0  2010        10              1     [0]
#         5   author 2              w1  2010        10              1     [0]
#         6   author 1              w1  2011        12              1     [2]
#         7   author 3              w3  2011        13              1     [3]
#         8   author 3              w4  2011        13              1     [3]
#         9   author 3              w5  2011        13              1     [3]
#         10  author 4              w5  2012        14              1     [4]
#         11  author 4              w3  2014        15              1     [5]

#         >>> keywords = Keywords(['author 1', 'author 2', 'author 3', 'w1', 'w3'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).summarize_by_term_per_term_per_year('Authors', 'Author Keywords', keywords=keywords)
#             Authors Author Keywords  Year  Cited by  Num Documents   ID
#         0  author 1              w1  2010        10              1  [0]
#         1  author 2              w1  2010        10              1  [0]
#         2  author 1              w1  2011        12              1  [2]
#         3  author 3              w3  2011        13              1  [3]

#         """

#         data = DataFrame(
#             self[[column_IDX, column_COL, "Year", "Cited by", "ID"]]
#         ).explode(column_IDX, sep_IDX)
#         data = DataFrame(data).explode(column_COL, sep_COL)
#         data["Num Documents"] = 1
#         result = data.groupby([column_IDX, column_COL, "Year"], as_index=False).agg(
#             {"Cited by": np.sum, "Num Documents": np.size}
#         )
#         result = result.assign(
#             ID=data.groupby([column_IDX, column_COL, "Year"])
#             .agg({"ID": list})
#             .reset_index()["ID"]
#         )
#         result["Cited by"] = result["Cited by"].map(lambda x: int(x))
#         if keywords is not None:
#             if keywords._patterns is None:
#                 keywords = keywords.compile()
#             result = result[result[column_IDX].map(lambda w: w in keywords)]
#             result = result[result[column_COL].map(lambda w: w in keywords)]
#         result.sort_values(
#             ["Year", column_IDX, column_COL,], ascending=True, inplace=True
#         )
#         return result.reset_index(drop=True)

#     def documents_by_terms_per_terms_per_year(
#         self, column_IDX, column_COL, sep_IDX=None, sep_COL=None, keywords=None
#     ):
#         """Computes the number of documents by term per term per year.

#         Args:
#             column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
#             sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
#             column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
#             sep_COL (str): Character used as internal separator for the elements in the column_COL.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#         ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
#         ...          "Cited by": list(range(10,16)),
#         ...          "ID": list(range(6)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors Author Keywords  Cited by  ID
#         0  2010  author 0;author 1;author 2           w0;w1        10   0
#         1  2010                    author 0              w0        11   1
#         2  2011                    author 1              w1        12   2
#         3  2011                    author 3        w5;w3;w4        13   3
#         4  2012                    author 4              w5        14   4
#         5  2014                    author 4              w3        15   5

#         >>> DataFrame(df).documents_by_terms_per_terms_per_year('Authors', 'Author Keywords')
#              Authors Author Keywords  Year  Num Documents      ID
#         0   author 0              w0  2010              2  [0, 1]
#         1   author 0              w1  2010              1     [0]
#         2   author 1              w0  2010              1     [0]
#         3   author 1              w1  2010              1     [0]
#         4   author 2              w0  2010              1     [0]
#         5   author 2              w1  2010              1     [0]
#         6   author 1              w1  2011              1     [2]
#         7   author 3              w3  2011              1     [3]
#         8   author 3              w4  2011              1     [3]
#         9   author 3              w5  2011              1     [3]
#         10  author 4              w5  2012              1     [4]
#         11  author 4              w3  2014              1     [5]

#         >>> keywords = Keywords(['author 1', 'author 2', 'author 3', 'w1', 'w3'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).documents_by_terms_per_terms_per_year('Authors', 'Author Keywords', keywords=keywords)
#             Authors Author Keywords  Year  Num Documents   ID
#         0  author 1              w1  2010              1  [0]
#         1  author 2              w1  2010              1  [0]
#         2  author 1              w1  2011              1  [2]
#         3  author 3              w3  2011              1  [3]

#         """

#         result = self.summarize_by_term_per_term_per_year(
#             column_IDX, column_COL, sep_IDX, sep_COL, keywords
#         )
#         result.pop("Cited by")
#         result.sort_values(
#             ["Year", column_IDX, column_COL],
#             ascending=[True, True, True],
#             inplace=True,
#         )
#         return result.reset_index(drop=True)

#     def citations_by_terms_per_terms_per_year(
#         self, column_IDX, column_COL, sep_IDX=None, sep_COL=None, keywords=None
#     ):
#         """Computes the number of citations by term per term per year.

#         Args:
#             column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
#             sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
#             column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
#             sep_COL (str): Character used as internal separator for the elements in the column_COL.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#         ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
#         ...          "Cited by": list(range(10,16)),
#         ...          "ID": list(range(6)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors Author Keywords  Cited by  ID
#         0  2010  author 0;author 1;author 2           w0;w1        10   0
#         1  2010                    author 0              w0        11   1
#         2  2011                    author 1              w1        12   2
#         3  2011                    author 3        w5;w3;w4        13   3
#         4  2012                    author 4              w5        14   4
#         5  2014                    author 4              w3        15   5

#         >>> DataFrame(df).citations_by_terms_per_terms_per_year('Authors', 'Author Keywords')
#              Authors Author Keywords  Year  Cited by      ID
#         0   author 0              w0  2010        21  [0, 1]
#         1   author 0              w1  2010        10     [0]
#         2   author 1              w0  2010        10     [0]
#         3   author 1              w1  2010        10     [0]
#         4   author 2              w0  2010        10     [0]
#         5   author 2              w1  2010        10     [0]
#         6   author 1              w1  2011        12     [2]
#         7   author 3              w3  2011        13     [3]
#         8   author 3              w4  2011        13     [3]
#         9   author 3              w5  2011        13     [3]
#         10  author 4              w5  2012        14     [4]
#         11  author 4              w3  2014        15     [5]

#         >>> keywords = Keywords(['author 1', 'author 2', 'author 3', 'w1', 'w3'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).citations_by_terms_per_terms_per_year('Authors', 'Author Keywords', keywords=keywords)
#             Authors Author Keywords  Year  Cited by   ID
#         0  author 1              w1  2010        10  [0]
#         1  author 2              w1  2010        10  [0]
#         2  author 1              w1  2011        12  [2]
#         3  author 3              w3  2011        13  [3]


#         """

#         result = self.summarize_by_term_per_term_per_year(
#             column_IDX, column_COL, sep_IDX, sep_COL, keywords
#         )
#         result.pop("Num Documents")
#         result.sort_values(
#             ["Year", column_IDX, column_COL],
#             ascending=[True, True, True],
#             inplace=True,
#         )
#         return result.reset_index(drop=True)


#     #
#     #
#     #  Occurrence
#     #
#     #


#     def occurrence_map(self, column, sep=None, minmax=None, keywords=None):
#         """Computes a occurrence between terms in a column.

#         Args:
#             column (str): the column to explode.
#             sep (str): Character used as internal separator for the elements in the column.
#             minmax (pair(number,number)): filter values by >=min,<=max.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             dictionary

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
#         >>> df = pd.DataFrame(
#         ...    {
#         ...       'Authors': x,
#         ...       'ID': list(range(len(x))),
#         ...    }
#         ... )
#         >>> df
#           Authors  ID
#         0       A   0
#         1       A   1
#         2     A;B   2
#         3       B   3
#         4   A;B;C   4
#         5       D   5
#         6     B;D   6

#         >>> DataFrame(df).occurrence_map(column='Authors')
#         {'terms': ['A', 'B', 'C', 'D'], 'docs': ['doc#0', 'doc#1', 'doc#2', 'doc#3', 'doc#4', 'doc#5'], 'edges': [('A', 'doc#0'), ('A', 'doc#1'), ('B', 'doc#1'), ('A', 'doc#2'), ('B', 'doc#2'), ('C', 'doc#2'), ('B', 'doc#3'), ('B', 'doc#4'), ('D', 'doc#4'), ('D', 'doc#5')], 'label_terms': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}, 'label_docs': {'doc#0': 2, 'doc#1': 1, 'doc#2': 1, 'doc#3': 1, 'doc#4': 1, 'doc#5': 1}}

#         >>> keywords = Keywords(['A', 'B'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).occurrence_map('Authors', keywords=keywords)
#         {'terms': ['A', 'B'], 'docs': ['doc#0', 'doc#1', 'doc#2', 'doc#3', 'doc#4'], 'edges': [('A', 'doc#0'), ('A', 'doc#1'), ('B', 'doc#1'), ('A', 'doc#2'), ('B', 'doc#2'), ('B', 'doc#3'), ('B', 'doc#4')], 'label_terms': {'A': 'A', 'B': 'B'}, 'label_docs': {'doc#0': 2, 'doc#1': 1, 'doc#2': 1, 'doc#3': 1, 'doc#4': 1}}


#         """
#         sep = ";" if sep is None and column in SCOPUS_COLS else sep

#         result = self[[column]].copy()
#         result["count"] = 1
#         result = result.groupby(column, as_index=False).agg({"count": np.sum})
#         if keywords is not None:
#             if keywords._patterns is None:
#                 keywords = keywords.compile()
#             if sep is not None:
#                 result = result[
#                     result[column].map(
#                         lambda x: any([e in keywords for e in x.split(sep)])
#                     )
#                 ]
#             else:
#                 result = result[result[column].map(lambda x: x in keywords)]

#         if sep is not None:
#             result[column] = result[column].map(
#                 lambda x: sorted(x.split(sep)) if isinstance(x, str) else x
#             )

#         result["doc-ID"] = ["doc#{:d}".format(i) for i in range(len(result))]
#         terms = result[[column]].copy()
#         terms.explode(column)
#         terms = [item for sublist in terms[column].tolist() for item in sublist]
#         terms = sorted(set(terms))
#         if keywords is not None:
#             terms = [x for x in terms if x in keywords]
#         docs = result["doc-ID"].tolist()
#         label_docs = {doc: label for doc, label in zip(docs, result["count"].tolist())}
#         label_terms = {t: t for t in terms}
#         edges = []
#         for field, docID in zip(result[column], result["doc-ID"]):
#             for item in field:
#                 if keywords is None or item in keywords:
#                     edges.append((item, docID))
#         return dict(
#             terms=terms,
#             docs=docs,
#             edges=edges,
#             label_terms=label_terms,
#             label_docs=label_docs,
#         )


#     def drop_duplicates(self, subset=None, keep="first"):
#         """Return DataFrame with duplicate rows removed.
#         """
#         result = self.drop_duplicates(sebset=subset, keep=keep)
#         return DataFrame(result)

#     def most_frequent(self, column, top_n=10, sep=None):
#         """Creates a group for most frequent items

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
#         >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
#         >>> df = pd.DataFrame(
#         ...    {
#         ...       'Authors': x,
#         ...       'Author Keywords': y,
#         ...       'Cited by': list(range(len(x))),
#         ...       'ID': list(range(len(x))),
#         ...    }
#         ... )
#         >>> df
#           Authors Author Keywords  Cited by  ID
#         0       A               a         0   0
#         1     A;B             a;b         1   1
#         2       B               b         2   2
#         3   A;B;C               c         3   3
#         4     B;D             c;d         4   4
#         5     A;B               d         5   5

#         >>> DataFrame(df).most_frequent('Authors', top_n=1)
#           Authors Author Keywords  Cited by  ID  top_1_Authors_freq
#         0       A               a         0   0               False
#         1     A;B             a;b         1   1                True
#         2       B               b         2   2                True
#         3   A;B;C               c         3   3                True
#         4     B;D             c;d         4   4                True
#         5     A;B               d         5   5                True

#         """
#         top = self.documents_by_term(column, sep=sep)[column].head(top_n)
#         items = Keywords().add_keywords(top)
#         colname = "top_{:d}_{:s}_freq".format(top_n, column.replace(" ", "_"))
#         df = DataFrame(self.copy())
#         sep = ";" if sep is None and column in SCOPUS_COLS else sep
#         if sep is not None:
#             df[colname] = self[column].map(
#                 lambda x: any([e in items for e in x.split(sep)])
#             )
#         else:
#             df[colname] = self[column].map(lambda x: x in items)
#         return DataFrame(df)

#     def most_cited_by(self, column, top_n=10, sep=None):
#         """Creates a group for most items cited by

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
#         >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
#         >>> df = pd.DataFrame(
#         ...    {
#         ...       'Authors': x,
#         ...       'Author Keywords': y,
#         ...       'Cited by': list(range(len(x))),
#         ...       'ID': list(range(len(x))),
#         ...    }
#         ... )
#         >>> df
#           Authors Author Keywords  Cited by  ID
#         0       A               a         0   0
#         1     A;B             a;b         1   1
#         2       B               b         2   2
#         3   A;B;C               c         3   3
#         4     B;D             c;d         4   4
#         5     A;B               d         5   5

#         >>> DataFrame(df).most_cited_by('Authors', top_n=1)
#           Authors Author Keywords  Cited by  ID  top_1_Authors_cited_by
#         0       A               a         0   0                   False
#         1     A;B             a;b         1   1                    True
#         2       B               b         2   2                    True
#         3   A;B;C               c         3   3                    True
#         4     B;D             c;d         4   4                    True
#         5     A;B               d         5   5                    True


#         """
#         top = self.citations_by_term(column, sep=sep)[column].head(top_n)
#         items = Keywords(keywords=top)
#         colname = "top_{:d}_{:s}_cited_by".format(top_n, column.replace(" ", "_"))
#         df = DataFrame(self.copy())
#         sep = ";" if sep is None and column in SCOPUS_COLS else sep
#         if sep is not None:
#             df[colname] = self[column].map(
#                 lambda x: any([e in items for e in x.split(sep)])
#             )
#         else:
#             df[colname] = self[column].map(lambda x: x in items)
#         return DataFrame(df)

#     def growth_indicators(self, column, sep=None, timewindow=2, keywords=None):
#         """Computes the average growth rate of a group of terms.

#         Args:
#             column (str): the column to explode.
#             sep (str): Character used as internal separator for the elements in the column.
#             timewindow (int): time window for analysis
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.


#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2013, 2014, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4,author 0;author 3,author 3;author 4".split(","),
#         ...          "Cited by": list(range(10,18)),
#         ...          "ID": list(range(8)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors  Cited by  ID
#         0  2010  author 0;author 1;author 2        10   0
#         1  2010                    author 0        11   1
#         2  2011                    author 1        12   2
#         3  2011                    author 3        13   3
#         4  2012                    author 4        14   4
#         5  2013                    author 4        15   5
#         6  2014           author 0;author 3        16   6
#         7  2014           author 3;author 4        17   7

#         >>> DataFrame(df).documents_by_term_per_year('Authors', as_matrix=True)
#               author 0  author 1  author 2  author 3  author 4
#         2010         2         1         1         0         0
#         2011         0         1         0         1         0
#         2012         0         0         0         0         1
#         2013         0         0         0         0         1
#         2014         1         0         0         2         1

#         >>> DataFrame(df).growth_indicators('Authors')
#             Authors       AGR  ADY   PDLY  Before 2013  Between 2013-2014
#         0  author 3  0.666667  1.0  12.50            1                  2
#         1  author 0  0.333333  0.5   6.25            2                  1
#         2  author 4  0.000000  1.0  12.50            1                  2

#         >>> keywords = Keywords(['author 3', 'author 4'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).growth_indicators('Authors', keywords=keywords)
#             Authors       AGR  ADY  PDLY  Before 2013  Between 2013-2014
#         0  author 3  0.666667  1.0  12.5            1                  2
#         1  author 4  0.000000  1.0  12.5            1                  2

#         """

#         def compute_agr():
#             result = self.documents_by_term_per_year(
#                 column=column, sep=sep, keywords=keywords
#             )
#             years_agr = sorted(set(result.Year))[-(timewindow + 1) :]
#             years_agr = [years_agr[0], years_agr[-1]]
#             result = result[result.Year.map(lambda w: w in years_agr)]
#             result.pop("ID")
#             result = pd.pivot_table(
#                 result,
#                 columns="Year",
#                 index=column,
#                 values="Num Documents",
#                 fill_value=0,
#             )
#             result["AGR"] = 0.0
#             result = result.assign(
#                 AGR=(result[years_agr[1]] - result[years_agr[0]]) / (timewindow + 1)
#             )
#             result.pop(years_agr[0])
#             result.pop(years_agr[1])
#             result = result.reset_index()
#             result.columns = list(result.columns)
#             result = result.sort_values(by=["AGR", column], ascending=False)
#             return result

#         def compute_ady():
#             result = self.documents_by_term_per_year(
#                 column=column, sep=sep, keywords=keywords
#             )
#             years_ady = sorted(set(result.Year))[-timewindow:]
#             result = result[result.Year.map(lambda w: w in years_ady)]
#             result = result.groupby([column], as_index=False).agg(
#                 {"Num Documents": np.sum}
#             )
#             result = result.set_index(column)
#             result = result.rename(columns={"Num Documents": "ADY"})
#             result["ADY"] = result.ADY.map(lambda w: w / timewindow)
#             return result.ADY

#         def compute_num_documents():
#             result = self.documents_by_term_per_year(
#                 column=column, sep=sep, keywords=keywords
#             )
#             years_between = sorted(set(result.Year))[-timewindow:]
#             years_before = sorted(set(result.Year))[0:-timewindow]
#             between = result[result.Year.map(lambda w: w in years_between)]
#             before = result[result.Year.map(lambda w: w in years_before)]
#             between = between.groupby([column], as_index=False).agg(
#                 {"Num Documents": np.sum}
#             )
#             between = between.rename(
#                 columns={
#                     "Num Documents": "Between {}-{}".format(
#                         years_between[0], years_between[-1]
#                     )
#                 }
#             )
#             before = before.groupby([column], as_index=False).agg(
#                 {"Num Documents": np.sum}
#             )
#             before = before.rename(
#                 columns={"Num Documents": "Before {}".format(years_between[0])}
#             )
#             result = pd.merge(before, between, on=column)
#             result = result.set_index(column)
#             return result

#         result = compute_agr()
#         result = result.set_index(column)
#         ady = compute_ady()
#         result.at[ady.index, "ADY"] = ady
#         result = result.assign(PDLY=round(result.ADY / len(self) * 100, 2))
#         num_docs = compute_num_documents()
#         result = pd.merge(result, num_docs, on=column)
#         result = result.reset_index()
#         return result

#     # def correspondence_matrix(
#     #     self,
#     #     column_IDX,
#     #     column_COL,
#     #     sep_IDX=None,
#     #     sep_COL=None,
#     #     as_matrix=False,
#     #     keywords=None,
#     # ):
#     #     """

#     #     """
#     #     result = self.co_occurrence(
#     #         column_IDX=column_IDX,
#     #         column_COL=column_COL,
#     #         sep_IDX=sep_IDX,
#     #         sep_COL=sep_COL,
#     #         as_matrix=True,
#     #         minmax=None,
#     #         keywords=keywords,
#     #     )

#     #     matrix = result.transpose().values
#     #     grand_total = np.sum(matrix)
#     #     correspondence_matrix = np.divide(matrix, grand_total)
#     #     row_totals = np.sum(correspondence_matrix, axis=1)
#     #     col_totals = np.sum(correspondence_matrix, axis=0)
#     #     independence_model = np.outer(row_totals, col_totals)
#     #     norm_correspondence_matrix = np.divide(
#     #         correspondence_matrix, row_totals[:, None]
#     #     )
#     #     distances = np.zeros(
#     #         (correspondence_matrix.shape[0], correspondence_matrix.shape[0])
#     #     )
#     #     norm_col_totals = np.sum(norm_correspondence_matrix, axis=0)
#     #     for row in range(correspondence_matrix.shape[0]):
#     #         distances[row] = np.sqrt(
#     #             np.sum(
#     #                 np.square(
#     #                     norm_correspondence_matrix - norm_correspondence_matrix[row]
#     #                 )
#     #                 / col_totals,
#     #                 axis=1,
#     #             )
#     #         )
#     #     std_residuals = np.divide(
#     #         (correspondence_matrix - independence_model), np.sqrt(independence_model)
#     #     )
#     #     u, s, vh = np.linalg.svd(std_residuals, full_matrices=False)
#     #     deltaR = np.diag(np.divide(1.0, np.sqrt(row_totals)))
#     #     rowScores = np.dot(np.dot(deltaR, u), np.diag(s))

#     #     return pd.DataFrame(
#     #         data=rowScores, columns=result.columns, index=result.columns
#     #     )


# # def __getitem__(self, key):
# #     return DataFrame(super().__getitem__(key))

# # def _getitem_bool_array(self, key):
# #     return DataFrame(super()._getitem_bool_array(key))

# # def _getitem_multilevel(self, key):
# #     return DataFrame(super()._getitem_multilevel(key))

# # def _get_column_array(self, i):
# #     return DataFrame(super()._get_column_array(i))

# # def assign(self, **kwargs):
# #     return DataFrame(super().assign(**kwargs))

# # def test(self):
# #     """

# #     >>> import pandas as pd
# #     >>> pd = pd.DataFrame(
# #     ...     {
# #     ...         "Authors": "author 3,author 1,author 0,author 2".split(","),
# #     ...         "Num Documents": [10, 5, 2, 1],
# #     ...         "ID": list(range(4)),
# #     ...     }
# #     ... )
# #     >>> pd = DataFrame(pd)
# #     >>> type(pd)
# #     <class 'techminer.dataframe.DataFrame'>

# #     >>> type(pd[[True, True, False, False]])
# #     <class 'pandas.core.frame.DataFrame'>

# #     # >>> pd[[True, True, False, False]]

# #     """

# #     pass


# if __name__ == "__main__":
#     import doctest

#     doctest.testmod()
