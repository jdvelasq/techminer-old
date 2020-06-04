
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
from techminer.explode import __explode
from techminer.plots import COLORMAPS
from techminer.keywords import Keywords

from techminer.explode import MULTIVALUED_COLS

def summary_co_occurrence(x, column, by=None, keywords=None):
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

    data = data[[by, column, "Cited by", "ID"]]
    data = data.dropna()
    data["Num Documents"] = 1
    data["pairs"] = [generate_pairs(a, b) for a, b in zip(data[by], data[column])]
    data = data[["pairs", "Num Documents", "Cited by", "ID"]]
    data = data.explode("pairs")
    result = data.groupby("pairs", as_index=False).agg(
        {"Cited by": np.sum, "Num Documents": np.sum, "ID": list}
    )
    result["Cited by"] = result["Cited by"].map(int)
    result[by] = result["pairs"].map(lambda x: x[0])
    result[column] = result["pairs"].map(lambda x: x[1])
    result.pop("pairs")
    result = result[[column, by, "Num Documents", "Cited by", "ID",]]
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
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
    #             new_names[column], new_names["Num Documents"],
    #         )
    #     }
    #     return new_names

    #
    result = summary_co_occurrence(x, column=column, by=by, keywords=keywords)
    if by is None or by == column:
        by = column + "_"
    result.pop("Cited by")
    #
    if as_matrix is False:
        result = result.sort_values(
            ["Num Documents", column, by], ascending=[False, True, True],
        )
        if min_value is not None and min_value > 0:
            result = result[result["Num Documents"] >= min_value]
        result = result.reset_index(drop=True)
        return result
    #
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Num Documents", index=by, columns=column, fill_value=0,
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


def occurrence(x, column, as_matrix=False, min_value=0, keywords=None):
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

COLUMNS = [
    "Author Keywords",
    "Authors",
    "Countries",
    "Country 1st",
    "Document type",
    "Index Keywords",
    "Institution 1st",
    "Institutions",
    "Keywords",
    "Source title",
]

def __body_0(x):
    #
    def server(**kwargs):
        #
        # Logic
        #
        term = kwargs["term"]
        by = kwargs["by"]
        cmap = kwargs["cmap"]
        min_value = kwargs["min_value"]
        #
        matrix = co_occurrence(
                x,
                column=term,
                by=by,
                as_matrix=True,
                min_value=min_value,
                keywords=None,
            )
        #
        if controls[3]['widget'].value > matrix.max().max():
            controls[3]['widget'].value = matrix.max().max()
        controls[3]['widget'].max = matrix.max().max()
        #
        output.clear_output()
        with output:
            if len(matrix.columns) < 51 and len(matrix.index) < 51:
                display(matrix.style.background_gradient(cmap=cmap))
            else:
                display(matrix)
                
    #
    # UI
    #
    controls = [
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Select(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "by",
            "desc": "By Term:",
            "widget": widgets.Select(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "min_value",
            "desc": "Lower value",
            "widget": widgets.IntSlider(
                    value=0,
                    min=0,
                    max=50,
                    step=1,
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        
    ]
    #
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
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH)),
        ]
    )


def app(df):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__body_0(df)]
    body.set_title(0, "Matrix")
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

