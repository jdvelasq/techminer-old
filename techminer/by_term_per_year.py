
"""
Analysis by Term per Year
==================================================================================================



"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.by_term import citations_by_term, documents_by_term
from techminer.explode import __explode
from techminer.keywords import Keywords
from techminer.plots import COLORMAPS


def summary_by_term_per_year(x, column, keywords=None):
    """Computes the number of documents and citations by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> summary_by_term_per_year(df, 'Authors')
        Authors  Year  Cited by  Num Documents      ID
    0  author 0  2010        21              2  [0, 1]
    1  author 1  2010        10              1     [0]
    2  author 2  2010        10              1     [0]
    3  author 1  2011        12              1     [2]
    4  author 3  2011        13              1     [3]
    5  author 4  2012        14              1     [4]
    6  author 4  2014        15              1     [5]

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> summary_by_term_per_year(df, 'Authors', keywords=keywords)
        Authors  Year  Cited by  Num Documents   ID
    0  author 1  2010        10              1  [0]
    1  author 2  2010        10              1  [0]
    2  author 1  2011        12              1  [2]
    3  author 3  2011        13              1  [3]

    """
    data = __explode(x[["Year", column, "Cited by", "ID"]], column)
    data["Num Documents"] = 1
    result = data.groupby([column, "Year"], as_index=False).agg(
        {"Cited by": np.sum, "Num Documents": np.size}
    )
    result = result.assign(
        ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
    )
    result["Cited by"] = result["Cited by"].map(lambda x: int(x))
    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        result = result[result[column].map(lambda w: w in keywords)]
    result.sort_values(
        ["Year", column], ascending=True, inplace=True, ignore_index=True,
    )
    return result


def documents_by_term_per_year(x, column, as_matrix=False, keywords=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> documents_by_term_per_year(df, 'Authors')
        Authors  Year  Num Documents      ID
    0  author 0  2010              2  [0, 1]
    1  author 1  2010              1     [0]
    2  author 2  2010              1     [0]
    3  author 1  2011              1     [2]
    4  author 3  2011              1     [3]
    5  author 4  2012              1     [4]
    6  author 4  2014              1     [5]

    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         2         1         1         0         0
    2011         0         1         0         1         0
    2012         0         0         0         0         1
    2014         0         0         0         0         1

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> documents_by_term_per_year(df, 'Authors', keywords=keywords, as_matrix=True)
          author 1  author 2  author 3
    2010         1         1         0
    2011         1         0         1

    """

    result = summary_by_term_per_year(x, column, keywords)
    result.pop("Cited by")
    result.sort_values(
        ["Year", "Num Documents", column], ascending=[True, False, True], inplace=True,
    )
    result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Num Documents", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    return result


def gant(x, column, keywords=None):
    """Computes the number of documents by term per year.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2011, 2011, 2012, 2015, 2012, 2016],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 3,author 4,author 4".split(","),
    ...          "Cited by": list(range(10,17)),
    ...          "ID": list(range(7)),
    ...     }
    ... )
    >>> documents_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         1         1         1         0         0
    2011         1         1         0         0         0
    2012         0         0         0         1         1
    2015         0         0         0         1         0
    2016         0         0         0         0         1

    >>> gant(df, 'Authors')
          author 0  author 1  author 2  author 3  author 4
    2010         1         1         1         0         0
    2011         1         1         0         0         0
    2012         0         0         0         1         1
    2013         0         0         0         1         1
    2014         0         0         0         1         1
    2015         0         0         0         1         1
    2016         0         0         0         0         1

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> gant(df, 'Authors', keywords=keywords)
          author 1  author 2  author 3
    2010         1         1         0
    2011         1         0         0
    2012         0         0         1
    2013         0         0         1
    2014         0         0         1
    2015         0         0         1

    """
    result = documents_by_term_per_year(
        x, column=column, as_matrix=True, keywords=keywords
    )
    years = [year for year in range(result.index.min(), result.index.max() + 1)]
    result = result.reindex(years, fill_value=0)
    matrix1 = result.copy()
    matrix1 = matrix1.cumsum()
    matrix1 = matrix1.applymap(lambda x: True if x > 0 else False)
    matrix2 = result.copy()
    matrix2 = matrix2.sort_index(ascending=False)
    matrix2 = matrix2.cumsum()
    matrix2 = matrix2.applymap(lambda x: True if x > 0 else False)
    matrix2 = matrix2.sort_index(ascending=True)
    result = matrix1.eq(matrix2)
    result = result.applymap(lambda x: 1 if x is True else 0)
    return result


def citations_by_term_per_year(x, column, as_matrix=False, keywords=None):
    """Computes the number of citations by term by year in a column.

    Args:
        column (str): the column to explode.
        as_matrix (bool): Results are returned as a matrix.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
    ...          "Cited by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Cited by  ID
    0  2010  author 0;author 1;author 2        10   0
    1  2010                    author 0        11   1
    2  2011                    author 1        12   2
    3  2011                    author 3        13   3
    4  2012                    author 4        14   4
    5  2014                    author 4        15   5

    >>> citations_by_term_per_year(df, 'Authors')
        Authors  Year  Cited by      ID
    0  author 0  2010        21  [0, 1]
    1  author 2  2010        10     [0]
    2  author 1  2010        10     [0]
    3  author 3  2011        13     [3]
    4  author 1  2011        12     [2]
    5  author 4  2012        14     [4]
    6  author 4  2014        15     [5]

    >>> citations_by_term_per_year(df, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010        21        10        10         0         0
    2011         0        12         0        13         0
    2012         0         0         0         0        14
    2014         0         0         0         0        15

    >>> keywords = Keywords(['author 1', 'author 2', 'author 3'])
    >>> keywords = keywords.compile()
    >>> citations_by_term_per_year(df, 'Authors', keywords=keywords)
        Authors  Year  Cited by   ID
    0  author 2  2010        10  [0]
    1  author 1  2010        10  [0]
    2  author 3  2011        13  [3]
    3  author 1  2011        12  [2]

    """
    result = summary_by_term_per_year(x, column, keywords)
    result.pop("Num Documents")
    result.sort_values(
        ["Year", "Cited by", column], ascending=[True, False, False], inplace=True,
    )
    result = result.reset_index(drop=True)
    if as_matrix == True:
        result = pd.pivot_table(
            result, values="Cited by", index="Year", columns=column, fill_value=0,
        )
        result.columns = result.columns.tolist()
        result.index = result.index.tolist()
    return result


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
            "widget": widgets.Select(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                disabled=False,
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "analysis_type",
            "desc": "Analysis type:",
            "widget": widgets.Dropdown(
                options=["Frequency", "Citation"],
                value="Frequency",
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "plot_type",
            "desc": "Plot type:",
            "widget": widgets.Dropdown(
                options=["Heatmap", "Gant"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.IntSlider(
                value=10,
                min=10,
                max=50,
                step=1,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
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
        term = kwargs["term"]
        analysis_type = kwargs["analysis_type"]
        plot_type = kwargs["plot_type"]
        cmap = kwargs["cmap"]
        top_n = kwargs["top_n"]
        #
        plots = {"Heatmap": plt.heatmap, "Gant": plt.gant}
        plot = plots[plot_type]
        #
        if analysis_type == "Frequency":
            top = documents_by_term(x, term).head(top_n)[term].tolist()
            matrix = documents_by_term_per_year(x, term, as_matrix=True)
        else:
            top = citations_by_term(x, term).head(top_n)[term].tolist()
            matrix = citations_by_term_per_year(x, term, as_matrix=True)
        matrix = matrix[top]
        output.clear_output()
        with output:
            if plot_type == "Heatmap":
                display(plot(matrix, cmap=cmap, figsize=FIGSIZE))
            if plot_type == "Gant":
                display(plot(matrix, figsize=FIGSIZE))

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
    body.children = [__body_0(df)]
    body.set_title(0, "Time Analysis")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Summary by Term per Year"
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
