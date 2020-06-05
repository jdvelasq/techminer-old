"""
Correlation Analysis
==================================================================================================



"""

import ipywidgets as widgets
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.by_term import summary_by_term
from techminer.co_occurrence import co_occurrence
from techminer.explode import MULTIVALUED_COLS, __explode
from techminer.keywords import Keywords
from techminer.maps import Map
from techminer.plots import COLORMAPS, chord_diagram


def compute_tfm(x, column, keywords=None):
    """Computes the term-frequency matrix for the terms in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
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

    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1

    >>> compute_tfm(df, 'Author Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> compute_tfm(df, 'Authors', keywords=keywords)
       A  B
    0  1  0
    1  1  1
    2  0  1
    3  1  1
    4  0  1

    """
    data = x[[column, "ID"]].copy()
    data["value"] = 1.0
    data = __explode(data, column)
    if keywords is not None:
        if isinstance(keywords, list):
            keywords = Keywords(keywords, ignore_case=False, full_match=True)
        if keywords._patterns is None:
            keywords = keywords.compile()
        data = data[data[column].map(lambda w: w in keywords)]
    result = pd.pivot_table(
        data=data, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)
    return result


def corr(
    x,
    column,
    by=None,
    method="pearson",
    #  min_link_value=-1,
    filter_by="Frequency",
    filter_value=0,
    cmap=None,
    as_matrix=True,
    keywords=None,
):
    """Computes cross-correlation among items in two different columns of the dataframe.

    Args:
        column_IDX (str): the first column.
        sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
        column_COL (str): the second column.
        sep_COL (str): Character used as internal separator for the elements in the column_COL.
        method (str): Available methods are:

            - pearson : Standard correlation coefficient.

            - kendall : Kendall Tau correlation coefficient.

            - spearman : Spearman rank correlation.

        as_matrix (bool): the result is reshaped by melt or not.
        minmax (pair(number,number)): filter values by >=min,<=max.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
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
    5     A;B               d         5   5


    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> compute_tfm(df, 'Author Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1
    5  0  0  0  1


    >>> corr(df, 'Authors', 'Author Keywords')
              A         B         C        D
    A  1.000000 -1.000000 -0.333333 -0.57735
    B -1.000000  1.000000  0.333333  0.57735
    C -0.333333  0.333333  1.000000  0.57735
    D -0.577350  0.577350  0.577350  1.00000

    >>> corr(df, 'Authors', 'Author Keywords', min_link_value=0)
              B         C        D
    B  1.000000  0.333333  0.57735
    C  0.333333  1.000000  0.57735
    D  0.577350  0.577350  1.00000

    >>> corr(df, 'Authors', 'Author Keywords', as_matrix=False)
       Authors Author Keywords     value
    0        A               A  1.000000
    1        B               A -1.000000
    2        C               A -0.333333
    3        D               A -0.577350
    4        A               B -1.000000
    5        B               B  1.000000
    6        C               B  0.333333
    7        D               B  0.577350
    8        A               C -0.333333
    9        B               C  0.333333
    10       C               C  1.000000
    11       D               C  0.577350
    12       A               D -0.577350
    13       B               D  0.577350
    14       C               D  0.577350
    15       D               D  1.000000

    >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> corr(df, 'Authors', 'Author Keywords', filter_by=keywords)
              A         B         C
    A  1.000000 -1.000000 -0.333333
    B -1.000000  1.000000  0.333333
    C -0.333333  0.333333  1.000000

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
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
    5     A;B               d         5   5

    >>> compute_tfm(df, column='Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0

    >>> corr(df, 'Authors')
              A         B         C         D
    A  1.000000 -0.316228  0.316228 -0.632456
    B -0.316228  1.000000  0.200000  0.200000
    C  0.316228  0.200000  1.000000 -0.200000
    D -0.632456  0.200000 -0.200000  1.000000

    >>> corr(df, 'Authors', as_matrix=False)
       Authors Authors_     value
    0        A        A  1.000000
    1        B        A -0.316228
    2        C        A  0.316228
    3        D        A -0.632456
    4        A        B -0.316228
    5        B        B  1.000000
    6        C        B  0.200000
    7        D        B  0.200000
    8        A        C  0.316228
    9        B        C  0.200000
    10       C        C  1.000000
    11       D        C -0.200000
    12       A        D -0.632456
    13       B        D  0.200000
    14       C        D -0.200000
    15       D        D  1.000000

    >>> corr(df, 'Author Keywords')
          a     b     c     d
    a  1.00  0.25 -0.50 -0.50
    b  0.25  1.00 -0.50 -0.50
    c -0.50 -0.50  1.00  0.25
    d -0.50 -0.50  0.25  1.00

    # >>> corr(df, 'Author Keywords', min_link_value=0.249)
    #       a     b     c     d
    # a  1.00  0.25 -0.50 -0.50
    # b  0.25  1.00 -0.50 -0.50
    # c -0.50 -0.50  1.00  0.25
    # d -0.50 -0.50  0.25  1.00


    # >>> corr(df, 'Author Keywords', min_link_value=1.0)
    #       c     d
    # c  1.00  0.25
    # d  0.25  1.00

    >>> keywords = Keywords(['A', 'B'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> corr(df, 'Authors', filter_by=keywords)
              A         B
    A  1.000000 -0.316228
    B -0.316228  1.000000


    """
    if by is None:
        by = column
    #
    column_filter = None
    if (filter_by == 0 or filter_by == "Frequency") and filter_value > 1:
        df = summary_by_term(x, column)
        if filter_value > df["Num Documents"].max():
            filter_value = df["Num Documents"].max()
        df = df[df["Num Documents"] >= filter_value]
        column_filter = df[column].tolist()
    if (filter_by == 1 or filter_by == "Cited by") and filter_value > 0:
        df = summary_by_term(x, column)
        if filter_value > df["Cited by"].max():
            filter_value = df["Cited by"].max()
        df = df[df["Cited by"] >= filter_value]
        column_filter = df[column].tolist()
    if column_filter is not None:
        column_filter = Keywords(column_filter, ignore_case=False, full_match=True)
    #
    if column == by:
        tfm = compute_tfm(x, column=column, keywords=column_filter)
    else:
        tfm = co_occurrence(
            x, column=column, by=by, as_matrix=True, keywords=column_filter,
        )
    result = tfm.corr(method=method)
    #
    if keywords is not None:
        keywords = keywords.compile()
        new_columns = [w for w in result.columns if w in keywords]
        new_index = [w for w in result.index if w in keywords]
        result = result.loc[new_index, new_columns]
    #
    if as_matrix is False:
        if column == by:
            result = (
                result.reset_index()
                .melt("index")
                .rename(columns={"index": column, "variable": column + "_"})
            )
        else:
            result = (
                result.reset_index()
                .melt("index")
                .rename(columns={"index": column, "variable": by})
            )
        return result
    result = result.sort_index(axis=0, ascending=True)
    result = result.sort_index(axis=1, ascending=True)
    return result


def corr_map(matrix, top_n_links=None, minval=0):
    """Computes the correlation map among items in a column of the dataframe.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        method (str): Available methods are:

            * pearson : Standard correlation coefficient.

            * kendall : Kendall Tau correlation coefficient.

            * spearman : Spearman rank correlation.

        minval (float): Minimum autocorrelation value to show links.
        top_n_links (int): Shows top n links.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;C', 'B', 'A;B;C', 'B;D', 'A;B', 'A;C']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd', 'c;d']
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
    1     A;C             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4
    5     A;B               d         5   5
    6     A;C             c;d         6   6


    >>> co_occurrence(df, 'Author Keywords', 'Authors', as_matrix=True)
       A  B  C  D
    a  2  0  1  0
    b  1  1  1  0
    c  2  2  2  1
    d  2  2  1  1

    >>> corr(df, 'Authors', 'Author Keywords')
              A         B         C         D
    A  1.000000  0.174078  0.333333  0.577350
    B  0.174078  1.000000  0.522233  0.904534
    C  0.333333  0.522233  1.000000  0.577350
    D  0.577350  0.904534  0.577350  1.000000

    >>> corr(df, 'Authors', 'Author Keywords')
              A         B         C         D
    A  1.000000  0.174078  0.333333  0.577350
    B  0.174078  1.000000  0.522233  0.904534
    C  0.333333  0.522233  1.000000  0.577350
    D  0.577350  0.904534  0.577350  1.000000

    >>> corr_map(df, 'Authors', 'Author Keywords')
    {'terms': ['A', 'B', 'C', 'D'], 'edges_75': None, 'edges_50': [('A', 'C')], 'edges_25': [('B', 'D')], 'other_edges': None}

    >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> corr_map(df, 'Authors', 'Author Keywords', keywords=keywords)
    {'terms': ['A', 'B', 'C'], 'edges_75': None, 'edges_50': [('A', 'C')], 'edges_25': None, 'other_edges': None}


    """

    if len(matrix.columns) > 50:
        return "Maximum number of nodex exceded!"

    terms = matrix.columns.tolist()

    n = len(matrix.columns)
    edges_75 = []
    edges_50 = []
    edges_25 = []
    other_edges = []

    if top_n_links is not None:
        values = matrix.to_numpy()
        top_value = []
        for icol in range(n):
            for irow in range(icol + 1, n):
                top_value.append(values[irow, icol])
        top_value = sorted(top_value, reverse=True)
        top_value = top_value[top_n_links - 1]
        if minval is not None:
            minval = max(minval, top_value)
        else:
            minval = top_value

    for icol in range(n):
        for irow in range(icol + 1, n):
            if minval is None or matrix[terms[icol]][terms[irow]] >= minval:
                if matrix[terms[icol]][terms[irow]] > 0.75:
                    edges_75.append((terms[icol], terms[irow]))
                elif matrix[terms[icol]][terms[irow]] > 0.50:
                    edges_50.append((terms[icol], terms[irow]))
                elif matrix[terms[icol]][terms[irow]] > 0.25:
                    edges_25.append((terms[icol], terms[irow]))
                elif matrix[terms[icol]][terms[irow]] > 0.0:
                    other_edges.append((terms[icol], terms[irow]))

    if len(edges_75) == 0:
        edges_75 = None
    if len(edges_50) == 0:
        edges_50 = None
    if len(edges_25) == 0:
        edges_25 = None
    if len(other_edges) == 0:
        other_edges = None

    return dict(
        terms=terms,
        edges_75=edges_75,
        edges_50=edges_50,
        edges_25=edges_25,
        other_edges=other_edges,
    )


#
#
#  Correlation Analysis
#
#

WIDGET_WIDTH = "200px"
LEFT_PANEL_HEIGHT = "650px"
RIGHT_PANEL_WIDTH = "870px"
FIGSIZE = (14, 10.0)
PANE_HEIGHTS = ["80px", "750px", 0]

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
        column = kwargs["term"]
        by = kwargs["by"]
        method = kwargs["method"]
        min_link_value = float(kwargs["min_link_value"].split(" ")[0])
        cmap = kwargs["cmap"]
        filter_by = kwargs["filter_by"]
        filter_value = int(kwargs["filter_value"].split()[0])
        view = kwargs["view"]
        sort_by = kwargs["sort_by"]
        #
        #
        #
        s = summary_by_term(x, column)
        if filter_by == "Frequency":
            s = s[[column, "Num Documents"]]
            n_columns = len(s[s["Num Documents"] > filter_value])
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[column].tolist(), s["Num Documents"].tolist())
            }
        if filter_by == "Cited by":
            s = s[[column, "Cited by"]]
            n_columns = len(s[s["Cited by"] > filter_value])
            new_names = {
                a: "{} [{:d}]".format(a, b)
                for a, b in zip(s[column].tolist(), s["Cited by"].tolist())
            }
        #
        a = s[s.columns[1]].value_counts().sort_index(ascending=False)
        a = a.cumsum()
        a = a.sort_index(ascending=True)
        current_value = controls[4]["widget"].value
        controls[4]["widget"].options = [
            "{:d} [{:d}]".format(idx, w) for w, idx in zip(a, a.index)
        ]
        if current_value not in controls[4]["widget"].options:
            controls[4]["widget"].value = controls[4]["widget"].options[0]
        #
        #
        #
        if n_columns > 50:
            output.clear_output()
            with output:
                display(widgets.HTML("<h3>Matrix exceeds the maximum shape</h3>"))
                return
        #
        #
        #
        matrix = corr(
            x,
            column=column,
            by=by,
            method=method,
            #  min_link_value=min_link_value,
            cmap=cmap,
            filter_by=filter_by,
            filter_value=filter_value,
            as_matrix=True,
            keywords=None,
        )
        #
        #
        #
        matrix = matrix.rename(columns=new_names, index=new_names)
        #
        output.clear_output()
        with output:
            if view == "Matrix":
                #
                # Sort order
                #
                if sort_by == "Frequency/Cited by asc.":
                    g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
                    names = sorted(matrix.columns, key=g, reverse=False)
                    matrix = matrix.loc[names, names]
                if sort_by == "Frequency/Cited by desc.":
                    g = lambda m: int(m[m.find("[") + 1 : m.find("]")])
                    names = sorted(matrix.columns, key=g, reverse=True)
                    matrix = matrix.loc[names, names]
                if sort_by == "Alphabetic asc.":
                    matrix = matrix.sort_index(axis=0, ascending=True).sort_index(
                        axis=1, ascending=True
                    )
                if sort_by == "Alphabetic desc.":
                    matrix = matrix.sort_index(axis=0, ascending=False).sort_index(
                        axis=1, ascending=False
                    )
                #
                # View
                #
                display(
                    matrix.style.format(
                        lambda q: "{:+4.3f}".format(q) if q >= min_link_value else ""
                    ).background_gradient(cmap=cmap)
                )
                #
            if view == "Correlation map":
                #
                display(
                    Map().correlation_map(
                        figsize=(10, 10), **corr_map(matrix, minval=min_link_value)
                    )
                )
                #
            if view == "Chord diagram":
                #
                display(chord_diagram(matrix, minval=min_link_value))
                #

    #
    # UI
    #
    controls = [
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "by",
            "desc": "By Term:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "method",
            "desc": "Method:",
            "widget": widgets.Dropdown(
                options=["pearson", "kendall", "spearman"],
                ensure_option=True,
                disabled=False,
                continuous_update=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "filter_by",
            "desc": "Filter by:",
            "widget": widgets.Dropdown(
                options=["Frequency", "Cited by"],
                disable=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "filter_value",
            "desc": "Filter value:",
            "widget": widgets.Dropdown(
                options=[str(i) for i in range(10)],
                disable=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "min_link_value",
            "desc": "Min link value:",
            "widget": widgets.Dropdown(
                options="-1.00 -0.25 0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875".split(
                    " "
                ),
                ensure_option=True,
                disabled=False,
                continuous_update=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Matrix", "Correlation map", "Chord diagram"],
                ensure_option=True,
                disabled=False,
                continuous_update=True,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        {
            "arg": "cmap",
            "desc": "Matrix colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
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
                disable=False,
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


#
#
# APP
#
#


def app(df):
    #
    body = widgets.Tab()
    body.children = [__body_0(df)]
    body.set_title(0, "Matrix")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Correlation Analysis"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
