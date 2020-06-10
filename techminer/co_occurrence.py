
"""
Co-occurrence Analysis
==================================================================================================



"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
# from techminer.explode import __explode
from techminer.plots import COLORMAPS
from techminer.keywords import Keywords
from techminer.by_term import summary_by_term
from techminer.explode import MULTIVALUED_COLS

def summary_co_occurrence(x, column, by=None, keywords=None, min_frequency=1, min_cited_by=0):
    """Summary occurrence and citations by terms in two different columns.

    Args:
        by (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the by.
        column (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author Keywords': y,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author Keywords  Cited by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4

    >>> summary_co_occurrence(df, column='Author Keywords', by='Authors')
      Author Keywords Authors  Num Documents  Cited by      ID
    0               a       A              2         1  [0, 1]
    1               a       B              1         1     [1]
    2               b       A              1         1     [1]
    3               b       B              2         3  [1, 2]
    4               c       A              1         3     [3]
    5               c       B              2         7  [3, 4]
    6               c       C              1         3     [3]
    7               c       D              1         4     [4]
    8               d       B              1         4     [4]
    9               d       D              1         4     [4]

    >>> keywords = Keywords(['B', 'C', 'a', 'b'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> summary_co_occurrence(df, 'Authors', 'Author Keywords', keywords=keywords)
      Authors Author Keywords  Num Documents  Cited by      ID
    0       B               a              1         1     [1]
    1       B               b              2         3  [1, 2]


    """

    def generate_pairs(w, v):
        if by in MULTIVALUED_COLS:
            w = [x.strip() for x in w.split(";")]
        else:
            w = [w]
        if column in MULTIVALUED_COLS:
            v = [x.strip() for x in v.split(";")]
        else:
            v = [v]
        result = []
        for idx0 in range(len(w)):
            for idx1 in range(len(v)):
                result.append((w[idx0], v[idx1]))
        return result

    #
    data = x.copy()
    if by is None or by == column:
        by = column + "_"
        data[by] = data[column].copy()

    data = data[[by, column, "Cited_by", "ID"]]
    data = data.dropna()
    data["Num_Documents"] = 1
    data["pairs"] = [generate_pairs(a, b) for a, b in zip(data[by], data[column])]
    data = data[["pairs", "Num_Documents", "Cited_by", "ID"]]
    data = data.explode("pairs")
    result = data.groupby("pairs", as_index=False).agg(
        {"Cited_by": np.sum, "Num_Documents": np.sum, "ID": list}
    )
    result["Cited_by"] = result["Cited_by"].map(int)
    result[by] = result["pairs"].map(lambda x: x[0])
    result[column] = result["pairs"].map(lambda x: x[1])
    result.pop("pairs")
    result = result[[column, by, "Num_Documents", "Cited_by", "ID",]]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[by].map(lambda w: w in keywords)]
        result = result[result[column].map(lambda w: w in keywords)]
    if min_frequency > 1 and min_cited_by > 0:
        df = summary_by_term(x, column)
        df = df[df['Num_Documents'] >= min_frequency]
        df = df[df['Cited_by'] >= min_cited_by]
        keywords = Keywords(df[column])
        if by[-1] != '_' and by[:-1] != column:
            df = summary_by_term(x, by)
            df = df[df['Num_Documents'] >= min_frequency]
            df = df[df['Cited_by'] >= min_cited_by]
            keywords += Keywords(df[by])
        keywords.compile()
        result = result[result[by].map(lambda w: w in keywords)]
        result = result[result[column].map(lambda w: w in keywords)]
    result = result.sort_values([column, by], ignore_index=True,)
    return result


def co_occurrence(
    x, column, by=None, as_matrix=False, min_value=0, keywords=None,
):
    """Computes the co-occurrence of two terms in different colums. The report adds
    the number of documents by term between brackets.

    Args:
        by (str): the column to explode. Their terms are used in the index of the result dataframe.
        sep_IDX (str): Character used as internal separator for the elements in the by.
        column (str): the column to explode. Their terms are used in the columns of the result dataframe.
        sep_COL (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author Keywords': y,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author Keywords  Cited by  ID
    0       A               a         0   0
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4

    >>> co_occurrence(df, column='Authors', by='Author Keywords')
      Authors Author Keywords  Num Documents      ID
    0       A               a              2  [0, 1]
    1       B               b              2  [1, 2]
    2       B               c              2  [3, 4]
    3       A               b              1     [1]
    4       A               c              1     [3]
    5       B               a              1     [1]
    6       B               d              1     [4]
    7       C               c              1     [3]
    8       D               c              1     [4]
    9       D               d              1     [4]

    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True)
       a  b  c  d
    A  2  1  1  0
    B  1  2  2  1
    C  0  0  1  0
    D  0  0  1  1

    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True, min_value=2)
       a  b  c
    A  2  1  1
    B  1  2  2

    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True, min_value=5)
       a  b  c
    A  2  1  1
    B  1  2  2

    >>> keywords = Keywords(['A', 'B', 'c', 'd'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> co_occurrence(df, column='Author Keywords', by='Authors', as_matrix=True, keywords=keywords)
       c  d
    A  1  0
    B  2  1

    

    """

    # def generate_dic(column, sep):
    #     new_names = documents_by_term(x, column)
    #     new_names = {
    #         term: "{:s} [{:d}]".format(term, docs_per_term)
    #         for term, docs_per_term in zip(
    #             new_names[column], new_names["Num_Documents"],
    #         )
    #     }
    #     return new_names

    #
    result = summary_co_occurrence(x, column=column, by=by, keywords=keywords)
    if by is None or by == column:
        by = column + "_"
    result.pop("Cited_by")
    #
    if as_matrix is False:
        result = result.sort_values(
            ["Num_Documents", column, by], ascending=[False, True, True],
        )
        if min_value is not None and min_value > 0:
            result = result[result["Num_Documents"] >= min_value]
        result = result.reset_index(drop=True)
        return result
    #
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Num_Documents", index=by, columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    if min_value is not None and min_value > 0:
        #
        a = result.max(axis=1)
        b = result.max(axis=0)
        a = a.sort_values(ascending=False)
        b = b.sort_values(ascending=False)
        min_value = (
            min(a.max(), b.max()) if min_value > min(a.max(), b.max()) else min_value
        )
        a = a[a >= min_value]
        b = b[b >= min_value]
        #
        result = result.loc[sorted(a.index), sorted(b.index)]
    return result


#
#
#


def summary_occurrence(x, column, keywords=None):
    """Summarize occurrence and citations by terms in a column of a dataframe.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors  Cited by  ID
    0       A         0   0
    1       A         1   1
    2     A;B         2   2
    3       B         3   3
    4   A;B;C         4   4
    5       D         5   5
    6     B;D         6   6

    >>> summary_occurrence(df, column='Authors')
       Authors Authors_  Num Documents  Cited by            ID
    0        A        A              4         7  [0, 1, 2, 4]
    1        A        B              2         6        [2, 4]
    2        A        C              1         4           [4]
    3        B        A              2         6        [2, 4]
    4        B        B              4        15  [2, 3, 4, 6]
    5        B        C              1         4           [4]
    6        B        D              1         6           [6]
    7        C        A              1         4           [4]
    8        C        B              1         4           [4]
    9        C        C              1         4           [4]
    10       D        B              1         6           [6]
    11       D        D              2        11        [5, 6]

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> summary_occurrence(df, 'Authors', keywords=keywords)
      Authors Authors_  Num Documents  Cited by            ID
    0       A        A              4         7  [0, 1, 2, 4]
    1       A        B              2         6        [2, 4]
    2       B        A              2         6        [2, 4]
    3       B        B              4        15  [2, 3, 4, 6]


    """
    return summary_co_occurrence(x=x, column=column, by=None, keywords=keywords)


def occurrence(x, column, as_matrix=False, normalization=None, min_value=0, keywords=None):
    """Computes the occurrence between the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (list, Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Cited by': list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors  Cited by  ID
    0       A         0   0
    1       A         1   1
    2     A;B         2   2
    3       B         3   3
    4   A;B;C         4   4
    5       D         5   5
    6     B;D         6   6

    >>> occurrence(df, column='Authors')
       Authors Authors_  Num Documents            ID
    0        A        A              4  [0, 1, 2, 4]
    1        B        B              4  [2, 3, 4, 6]
    2        A        B              2        [2, 4]
    3        B        A              2        [2, 4]
    4        D        D              2        [5, 6]
    5        A        C              1           [4]
    6        B        C              1           [4]
    7        B        D              1           [6]
    8        C        A              1           [4]
    9        C        B              1           [4]
    10       C        C              1           [4]
    11       D        B              1           [6]

    >>> occurrence(df, column='Authors', as_matrix=True)
       A  B  C  D
    A  4  2  1  0
    B  2  4  1  1
    C  1  1  1  0
    D  0  1  0  2

    >>> occurrence(df, column='Authors', min_value=2, as_matrix=True)
       A  B  D
    A  4  2  0
    B  2  4  1
    D  0  1  2

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> occurrence(df, 'Authors', as_matrix=True, keywords=keywords)
       A  B
    A  4  2
    B  2  4

    """
    return co_occurrence(
        x,
        column=column,
        by=None,
        as_matrix=as_matrix,
        min_value=min_value,
        keywords=keywords,
    )

def normalize_matrix(matrix, normalization=None):
    matrix = matrix.applymap(lambda w: float(w))
    m = matrix.copy()
    if normalization == 'association':    
        for col in m.columns:
            for row in m.index:
                matrix.at[row, col] = m.at[row, col] / (m.loc[row, row] * m.at[col, col])
    if normalization == 'inclusion':
        for col in m.columns:
            for row in m.index:
                matrix.at[row, col] = m.at[row, col] / min(m.loc[row, row], m.at[col, col])
    if normalization == 'jaccard':
        for col in m.columns:
            for row in m.index:
                matrix.at[row, col] = m.at[row, col] /(m.loc[row, row] + m.at[col, col] - m.at[row, col])
    if normalization == 'salton':
        for col in m.columns:
            for row in m.index:
                matrix.at[row, col] = m.at[row, col] / np.sqrt((m.loc[row, row] * m.at[col, col]))
    return matrix




# #     def occurrence_map(self, column, sep=None, minmax=None, keywords=None):
# #         """Computes a occurrence between terms in a column.

# #         Args:
# #             column (str): the column to explode.
# #             sep (str): Character used as internal separator for the elements in the column.
# #             minmax (pair(number,number)): filter values by >=min,<=max.
# #             keywords (Keywords): filter the result using the specified Keywords object.

# #         Returns:
# #             dictionary

# #         Examples
# #         ----------------------------------------------------------------------------------------------

# #         >>> import pandas as pd
# #         >>> x = [ 'A', 'A', 'A;B', 'B', 'A;B;C', 'D', 'B;D']
# #         >>> df = pd.DataFrame(
# #         ...    {
# #         ...       'Authors': x,
# #         ...       'ID': list(range(len(x))),
# #         ...    }
# #         ... )
# #         >>> df
# #           Authors  ID
# #         0       A   0
# #         1       A   1
# #         2     A;B   2
# #         3       B   3
# #         4   A;B;C   4
# #         5       D   5
# #         6     B;D   6

# #         >>> DataFrame(df).occurrence_map(column='Authors')
# #         {'terms': ['A', 'B', 'C', 'D'], 'docs': ['doc#0', 'doc#1', 'doc#2', 'doc#3', 'doc#4', 'doc#5'], 'edges': [('A', 'doc#0'), ('A', 'doc#1'), ('B', 'doc#1'), ('A', 'doc#2'), ('B', 'doc#2'), ('C', 'doc#2'), ('B', 'doc#3'), ('B', 'doc#4'), ('D', 'doc#4'), ('D', 'doc#5')], 'label_terms': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}, 'label_docs': {'doc#0': 2, 'doc#1': 1, 'doc#2': 1, 'doc#3': 1, 'doc#4': 1, 'doc#5': 1}}

# #         >>> keywords = Keywords(['A', 'B'])
# #         >>> keywords = keywords.compile()
# #         >>> DataFrame(df).occurrence_map('Authors', keywords=keywords)
# #         {'terms': ['A', 'B'], 'docs': ['doc#0', 'doc#1', 'doc#2', 'doc#3', 'doc#4'], 'edges': [('A', 'doc#0'), ('A', 'doc#1'), ('B', 'doc#1'), ('A', 'doc#2'), ('B', 'doc#2'), ('B', 'doc#3'), ('B', 'doc#4')], 'label_terms': {'A': 'A', 'B': 'B'}, 'label_docs': {'doc#0': 2, 'doc#1': 1, 'doc#2': 1, 'doc#3': 1, 'doc#4': 1}}


# #         """
# #         sep = ";" if sep is None and column in SCOPUS_COLS else sep

# #         result = self[[column]].copy()
# #         result["count"] = 1
# #         result = result.groupby(column, as_index=False).agg({"count": np.sum})
# #         if keywords is not None:
# #             if keywords._patterns is None:
# #                 keywords = keywords.compile()
# #             if sep is not None:
# #                 result = result[
# #                     result[column].map(
# #                         lambda x: any([e in keywords for e in x.split(sep)])
# #                     )
# #                 ]
# #             else:
# #                 result = result[result[column].map(lambda x: x in keywords)]

# #         if sep is not None:
# #             result[column] = result[column].map(
# #                 lambda x: sorted(x.split(sep)) if isinstance(x, str) else x
# #             )

# #         result["doc-ID"] = ["doc#{:d}".format(i) for i in range(len(result))]
# #         terms = result[[column]].copy()
# #         terms.explode(column)
# #         terms = [item for sublist in terms[column].tolist() for item in sublist]
# #         terms = sorted(set(terms))
# #         if keywords is not None:
# #             terms = [x for x in terms if x in keywords]
# #         docs = result["doc-ID"].tolist()
# #         label_docs = {doc: label for doc, label in zip(docs, result["count"].tolist())}
# #         label_terms = {t: t for t in terms}
# #         edges = []
# #         for field, docID in zip(result[column], result["doc-ID"]):
# #             for item in field:
# #                 if keywords is None or item in keywords:
# #                     edges.append((item, docID))
# #         return dict(
# #             terms=terms,
# #             docs=docs,
# #             edges=edges,
# #             label_terms=label_terms,
# #             label_docs=label_docs,
# #         )




# #     # def correspondence_matrix(
# #     #     self,
# #     #     column_IDX,
# #     #     column_COL,
# #     #     sep_IDX=None,
# #     #     sep_COL=None,
# #     #     as_matrix=False,
# #     #     keywords=None,
# #     # ):
# #     #     """

# #     #     """
# #     #     result = self.co_occurrence(
# #     #         column_IDX=column_IDX,
# #     #         column_COL=column_COL,
# #     #         sep_IDX=sep_IDX,
# #     #         sep_COL=sep_COL,
# #     #         as_matrix=True,
# #     #         minmax=None,
# #     #         keywords=keywords,
# #     #     )

# #     #     matrix = result.transpose().values
# #     #     grand_total = np.sum(matrix)
# #     #     correspondence_matrix = np.divide(matrix, grand_total)
# #     #     row_totals = np.sum(correspondence_matrix, axis=1)
# #     #     col_totals = np.sum(correspondence_matrix, axis=0)
# #     #     independence_model = np.outer(row_totals, col_totals)
# #     #     norm_correspondence_matrix = np.divide(
# #     #         correspondence_matrix, row_totals[:, None]
# #     #     )
# #     #     distances = np.zeros(
# #     #         (correspondence_matrix.shape[0], correspondence_matrix.shape[0])
# #     #     )
# #     #     norm_col_totals = np.sum(norm_correspondence_matrix, axis=0)
# #     #     for row in range(correspondence_matrix.shape[0]):
# #     #         distances[row] = np.sqrt(
# #     #             np.sum(
# #     #                 np.square(
# #     #                     norm_correspondence_matrix - norm_correspondence_matrix[row]
# #     #                 )
# #     #                 / col_totals,
# #     #                 axis=1,
# #     #             )
# #     #         )
# #     #     std_residuals = np.divide(
# #     #         (correspondence_matrix - independence_model), np.sqrt(independence_model)
# #     #     )
# #     #     u, s, vh = np.linalg.svd(std_residuals, full_matrices=False)
# #     #     deltaR = np.diag(np.divide(1.0, np.sqrt(row_totals)))
# #     #     rowScores = np.dot(np.dot(deltaR, u), np.diag(s))

# #     #     return pd.DataFrame(
# #     #         data=rowScores, columns=result.columns, index=result.columns
# #     #     )


# # def relationship(x, y):
# #     sxy = sum([a * b * min(a, b) for a, b in zip(x, y)])
# #     a = math.sqrt(sum(x))
# #     b = math.sqrt(sum(y))
# #     return sxy / (a * b)


##
##
## APP
##
##

WIDGET_WIDTH = "200px"
LEFT_PANEL_HEIGHT = "588px"
RIGHT_PANEL_WIDTH = "870px"
FIGSIZE = (14, 10.0)
PANE_HEIGHTS = ["80px", "650px", 0]

#
#
#  Co-occurrence Matrix
#
#

def __TAB0__(x):
    COLUMNS = [
        "Author_Keywords",
        "Author_Keywords_CL",
        "Authors",
        "Countries",
        "Country_1st_Author",
        "Document_type",
        "Index_Keywords",
        "Index_Keywords_CL",
        "Institution_1st_Author",
        "Institutions",
        "Source title",
    ]
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "by",
            "desc": "By Term:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "plot_type",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Heatmap", "Bubble plot", "Table"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },        
        # 3
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "min_value",
            "desc": "Min occurrence value:",
            "widget": widgets.Dropdown(
                options=['1'],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=[
                    "Alphabetic asc.",
                    "Alphabetic desc.",
                    "Frequency/Cited by asc.",
                    "Frequency/Cited by desc.",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        # Logic
        #
        term = kwargs["term"]
        by = kwargs["by"]
        plot_type = kwargs["plot_type"]
        cmap = kwargs["cmap"]
        min_value = int(kwargs["min_value"])
        sort_by = kwargs["sort_by"]
        #
        # 
        #
        matrix = co_occurrence(
                x,
                column=term,
                by=by,
                as_matrix=False,
                min_value=0,
                keywords=None,
            )
        #
        options = matrix['Num_Documents'].unique().tolist()
        options = sorted(options)
        options = [str(w) for w in options]
        current_selection = controls[4]["widget"].value
        controls[4]["widget"].options = options
        #
        if min_value > matrix['Num_Documents'].max():
            min_value = 0
            controls[4]["widget"].value = controls[4]["widget"].options[0]
        if current_selection not in controls[4]["widget"].options:
            min_value = 0
            controls[4]["widget"].value = controls[4]["widget"].options[0]
        #
        if term == by:
            by = by + '_'
        #
        matrix = pd.pivot_table(
            matrix, values="Num_Documents", index=by, columns=term, fill_value=0,
        )
        matrix.columns = matrix.columns.tolist()
        matrix.index = matrix.index.tolist()
        matrix = matrix.loc[sorted(matrix.index), sorted(matrix.columns)]
        #
        if min_value > 0:
            #
            a = matrix.max(axis=1)
            b = matrix.max(axis=0)
            a = a.sort_values(ascending=False)
            b = b.sort_values(ascending=False)
            a = a[a >= min_value]
            b = b[b >= min_value]
            matrix = matrix.loc[sorted(a.index), sorted(b.index)]
        #
        #
        if max(len(matrix.index), len(matrix.columns)) > 60:
            output.clear_output()
            with output:
                display(widgets.HTML("<h3>Matrix exceeds the maximum shape</h3>"))
                return
        #
        #
        #
        s = summary_by_term(x, term)
        new_names = {
            a: "{} [{:d}]".format(a, b)
            for a, b in zip(s[term].tolist(), s["Num_Documents"].tolist())
        }
        matrix = matrix.rename(columns=new_names)
        #
        if term == by[:-1]:
            matrix = matrix.rename(index=new_names)
        else:
            s = summary_by_term(x, by)
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[by].tolist(), s["Num_Documents"].tolist())
            }
            matrix = matrix.rename(index=new_names)
        #
        # Sort order
        #
        g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
        if sort_by == "Frequency/Cited by asc.":
            col_names = sorted(matrix.columns, key=g, reverse=False)
            row_names = sorted(matrix.index, key=g, reverse=False)
            matrix = matrix.loc[row_names, col_names]
        if sort_by == "Frequency/Cited by desc.":
            col_names = sorted(matrix.columns, key=g, reverse=True)
            row_names = sorted(matrix.index, key=g, reverse=True)
            matrix = matrix.loc[row_names, col_names]
        if sort_by == "Alphabetic asc.":
            matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
                axis=1, ascending=True
            )
        if sort_by == "Alphabetic desc.":
            matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
                axis=1, ascending=False
            )
        #
        output.clear_output()
        with output:
            #
            # View
            #
            if plot_type == 'Table':
                with pd.option_context("display.max_columns", 60, "display.max_rows", 60):
                    display(matrix.style.format(
                                lambda q: "{:d}".format(q) if q >= min_value else ""
                            ).background_gradient(cmap=cmap, axis=None))
            if plot_type == 'Heatmap':
                display(plt.heatmap(matrix, cmap=cmap, figsize=(14, 8.5)))
            if plot_type == 'Bubble plot':
                display(plt.bubble(matrix.transpose(), axis=0, cmap=cmap, figsize=(14, 8.5)))
            
    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
        ]
    )


#
#
#  Occurrence Network
#
#

def __TAB1__(x):
    COLUMNS = [
        "Author_Keywords",
        "Author_Keywords_CL",
        "Authors",
        "Countries",
        "Country_1st_Author",
        "Document_type",
        "Index_Keywords",
        "Index_Keywords_CL",
        "Institution_1st_Author",
        "Institutions",
        "Source title",
    ]
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "normalization",
            "desc": "Normalization:",
            "widget": widgets.Dropdown(
                options=['None', 'association', 'inclusion', 'jaccard', 'salton'],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "plot_type",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Heatmap", "Bubble plot", "Network plot", "Table"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },        
        # 3
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "min_value",
            "desc": "Min occurrence value:",
            "widget": widgets.Dropdown(
                options=['1'],
                ensure_option=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=[
                    "Alphabetic asc.",
                    "Alphabetic desc.",
                    "Frequency/Cited by asc.",
                    "Frequency/Cited by desc.",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "layout",
            "desc": "Map layout:",
            "widget": widgets.Dropdown(
                options=[
                    "Circular",
                    "Kamada Kawai",
                    "Planar",
                    "Random",
                    "Spectral",
                    "Spring",
                    "Shell",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        # Logic
        #
        term = kwargs["term"]
        normalization = kwargs["normalization"]
        plot_type = kwargs["plot_type"]
        cmap = kwargs["cmap"]
        min_value = int(kwargs["min_value"])
        sort_by = kwargs["sort_by"]
        layout = kwargs["layout"]
        #
        # 
        #
        matrix = occurrence(
                x,
                column=term,
                as_matrix=False,
                min_value=0,
                keywords=None,
            )
        #
        options = matrix['Num_Documents'].unique().tolist()
        options = sorted(options)
        options = [str(w) for w in options]
        current_selection = controls[4]["widget"].value
        controls[4]["widget"].options = options
        #
        if min_value > matrix['Num_Documents'].max():
            min_value = 0
            controls[4]["widget"].value = controls[4]["widget"].options[0]
        if current_selection not in controls[4]["widget"].options:
            min_value = 0
            controls[4]["widget"].value = controls[4]["widget"].options[0]
        #
        by = term + '_'
        #
        matrix = pd.pivot_table(
            matrix, values="Num_Documents", index=by, columns=term, fill_value=0,
        )
        matrix.columns = matrix.columns.tolist()
        matrix.index = matrix.index.tolist()
        matrix = matrix.loc[sorted(matrix.index), sorted(matrix.columns)]
        #
        if min_value > 0:
            #
            a = matrix.max(axis=1)
            b = matrix.max(axis=0)
            a = a.sort_values(ascending=False)
            b = b.sort_values(ascending=False)
            a = a[a >= min_value]
            b = b[b >= min_value]
            matrix = matrix.loc[sorted(a.index), sorted(b.index)]
        #
        #
        if max(len(matrix.index), len(matrix.columns)) > 60:
            output.clear_output()
            with output:
                display(widgets.HTML("<h3>Matrix exceeds the maximum shape</h3>"))
                return
        #
        s = summary_by_term(x, term)
        new_names = {
            a: "{} [{:d}]".format(a, b)
            for a, b in zip(s[term].tolist(), s["Num_Documents"].tolist())
        }
        matrix = matrix.rename(columns=new_names)
        #
        if term == by[:-1]:
            matrix = matrix.rename(index=new_names)
        else:
            s = summary_by_term(x, by)
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[by].tolist(), s["Num_Documents"].tolist())
            }
            matrix = matrix.rename(index=new_names)
        #
        #
        #
        matrix = normalize_matrix(matrix, normalization)
        #
        # Sort order
        #
        g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
        if sort_by == "Frequency/Cited by asc.":
            col_names = sorted(matrix.columns, key=g, reverse=False)
            row_names = sorted(matrix.index, key=g, reverse=False)
            matrix = matrix.loc[row_names, col_names]
        if sort_by == "Frequency/Cited by desc.":
            col_names = sorted(matrix.columns, key=g, reverse=True)
            row_names = sorted(matrix.index, key=g, reverse=True)
            matrix = matrix.loc[row_names, col_names]
        if sort_by == "Alphabetic asc.":
            matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
                axis=1, ascending=True
            )
        if sort_by == "Alphabetic desc.":
            matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
                axis=1, ascending=False
            )
        output.clear_output()
        with output:
            #
            # View
            #
            if plot_type == 'Table':
                with pd.option_context("display.max_columns", 60, "display.max_rows", 60):
                    if normalization == 'None':
                        display(matrix.style.format(
                                    lambda q: "{:g}".format(q) 
                                ).background_gradient(cmap=cmap, axis=None))
                    else:
                        display(matrix.style.format(
                                    lambda q: "{:3.2f}".format(q) 
                                ).background_gradient(cmap=cmap, axis=None))
            if plot_type == 'Heatmap':
                display(plt.heatmap(matrix, cmap=cmap, figsize=(14, 8.5)))
            if plot_type == 'Bubble plot':
                display(plt.bubble(matrix.transpose(), axis=0, cmap=cmap, figsize=(14, 8.5)))
            if plot_type == "Network plot":
                display(occurrence_map(
                    matrix, layout=layout, cmap=cmap, figsize=(14, 8.5)
                ))
            
    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
        ]
    )


def app(df):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__TAB0__(df), __TAB1__(df)]
    body.set_title(0, "Co-occurrence")
    body.set_title(1, "Occurrence")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Co-occurrence Analysis"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )



import matplotlib.pyplot as pyplot
import networkx as nx

def occurrence_map(
    matrix, layout="Kamada Kawai", cmap="Greys", figsize=(17, 12)
):
    """Computes the occurrence map directly using networkx.
    """

    if len(matrix.columns) > 50:
        return "Maximum number of nodex exceded!"

    #
    # Data preparation
    #
    terms = matrix.columns.tolist()

    #
    # Node sizes
    #
    node_sizes = [int(w[w.find("[") + 1 : w.find("]")]) for w in terms if "[" in w]
    if len(node_sizes) == 0:
        node_sizes = [10] * len(terms)
    else:
        max_size = max(node_sizes)
        min_size = min(node_sizes)
        if min_size == max_size:
            node_sizes = [300] * len(terms)
        else:
            node_sizes = [
                300 + int(1000 * (w - min_size) / (max_size - min_size))
                for w in node_sizes
            ]

    #
    # Node colors
    #
    cmap = pyplot.cm.get_cmap(cmap)
    node_colors = [
        cmap(0.2 + 0.75 * node_sizes[i] / max(node_sizes))
        for i in range(len(node_sizes))
    ]

    #
    # Remove [...] from text
    #
    terms = [w[: w.find("[")].strip() if "[" in w else w for w in terms]
    matrix.columns = terms
    matrix.index = terms

    #
    # Draw the network
    #
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    draw_dict = {
        "Circular": nx.draw_circular,
        "Kamada Kawai": nx.draw_kamada_kawai,
        "Planar": nx.draw_planar,
        "Random": nx.draw_random,
        "Spectral": nx.draw_spectral,
        "Spring": nx.draw_spring,
        "Shell": nx.draw_shell,
    }
    draw = draw_dict[layout]

    G = nx.Graph(ax=ax)
    G.clear()

    #
    # network nodes
    #
    G.add_nodes_from(terms)

    #
    # network edges
    #
    n = len(matrix.columns)
    max_width = 0
    for icol in range(n-1):
        for irow in range(icol + 1, n):
            link = matrix.at[matrix.columns[irow], matrix.columns[icol]]
            if  link > 0:
                G.add_edge(terms[icol], terms[irow], width=link)
                if max_width < link:
                    max_width = link
    #
    # Draw nodes
    #
    first_time = True
    for e in G.edges.data():
        a, b, width = e
        edge = [(a, b)]
        width = 0.2 + 4.0 * width['width'] / max_width
        draw(
            G,
            ax=ax,
            edgelist=edge,
            width=width,
            edge_color="k",
            with_labels=first_time,
            font_weight="bold",
            node_color=node_colors,
            node_size=node_sizes,
            bbox=dict(facecolor="white", alpha=1.0),
            font_size=10,
            horizontalalignment="left",
            verticalalignment="baseline",
        )
        first_time = False

    #
    

    #
    # Figure size
    #
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(
        xlim[0] - 0.15 * (xlim[1] - xlim[0]), xlim[1] + 0.15 * (xlim[1] - xlim[0])
    )
    ax.set_ylim(
        ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0])
    )

    return fig

