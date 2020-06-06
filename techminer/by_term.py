
"""
Analysis by Term
==================================================================================================



"""
import ipywidgets as widgets
import pandas as pd
import numpy as np
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.explode import __explode
from techminer.keywords import Keywords
from techminer.plots import COLORMAPS


def summary_by_term(x, column, keywords=None):
    """Summarize the number of documents and citations by term in a dataframe.

    Args:
        column (str): the column to explode.
        keywords (int, list): filter the results.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> summary_by_term(x, 'Authors')
        Authors  Num Documents  Cited by      ID
    0  author 0              2        21  [0, 1]
    1  author 1              2        22  [0, 2]
    2  author 2              1        10     [0]
    3  author 3              1        13     [3]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> summary_by_term(x, 'Authors', keywords=keywords)
        Authors  Num Documents  Cited by      ID
    0  author 1              2        22  [0, 2]
    1  author 2              1        10     [0]

    """
    x = x.copy()
    x = __explode(x[[column, "Cited by", "ID"]], column)
    x["Num Documents"] = 1
    result = x.groupby(column, as_index=False).agg(
        {"Num Documents": np.size, "Cited by": np.sum}
    )
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result["Cited by"] = result["Cited by"].map(lambda x: int(x))
    if keywords is not None:
        result = result[result[column].map(lambda w: w in keywords)]
    result.sort_values(
        [column, "Num Documents", "Cited by"],
        ascending=[True, False, False],
        inplace=True,
        ignore_index=True,
    )
    return result


def documents_by_term(x, column, keywords=None):
    """Computes the number of documents per term in a given column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> documents_by_term(x, 'Authors')
        Authors  Num Documents      ID
    0  author 0              2  [0, 1]
    1  author 1              2  [0, 2]
    2  author 2              1     [0]
    3  author 3              1     [3]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> documents_by_term(x, 'Authors', keywords=keywords)
        Authors  Num Documents      ID
    0  author 1              2  [0, 2]
    1  author 2              1     [0]

    """

    result = summary_by_term(x, column, keywords)
    result.pop("Cited by")
    result.sort_values(
        ["Num Documents", column],
        ascending=[False, True],
        inplace=True,
        ignore_index=True,
    )
    return result


def citations_by_term(x, column, keywords=None):
    """Computes the number of citations by item in a column.

    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> citations_by_term(x, 'Authors')
        Authors  Cited by      ID
    0  author 1        22  [0, 2]
    1  author 0        21  [0, 1]
    2  author 3        13     [3]
    3  author 2        10     [0]

    >>> keywords = Keywords(['author 1', 'author 2'])
    >>> keywords = keywords.compile()
    >>> citations_by_term(x, 'Authors', keywords=keywords)
        Authors  Cited by      ID
    0  author 1        22  [0, 2]
    1  author 2        10     [0]


    """
    result = summary_by_term(x, column, keywords)
    result.pop("Num Documents")
    result.sort_values(
        ["Cited by", column], ascending=[False, True], inplace=True, ignore_index=True,
    )
    return result



def most_cited_documents(x):
    """ Returns the cited documents.

    Results:
        pandas.DataFrame

    """
    result = x.sort_values(by="Cited by", ascending=False)[
        ["Title", "Authors", "Year", "Cited by", "ID"]
    ]
    result["Cited by"] = result["Cited by"].map(
        lambda w: int(w) if pd.isna(w) is False else 0
    )
    return result


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


##
##
##  APP
##
##

WIDGET_WIDTH = "200px"
LEFT_PANEL_HEIGHT = "588px"
RIGHT_PANEL_WIDTH = "870px"
FIGSIZE = (14, 8.2)
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

##
##
##  Panel 0
##
##
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
        # 2
        {
            "arg": "plot_type",
            "desc": "Plot type:",
            "widget": widgets.Dropdown(
                    options=["bar", "barh", "pie"],
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 3
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
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
        term = kwargs['term']
        cmap = kwargs['cmap']
        analysis_type = kwargs['analysis_type']
        top_n = kwargs['top_n']
        plot_type = kwargs['plot_type']
        #
        plots = {"bar": plt.bar, "barh": plt.barh, "pie": plt.pie}
        #
        df = summary_by_term(x, term)
        if analysis_type == "Frequency":
            df = df.sort_values(
                ["Num Documents", "Cited by", term], ascending=False
            )
            df = df[[term, "Num Documents"]].head(top_n)
        else:
            df = df.sort_values(
                ["Cited by", "Num Documents", term], ascending=False
            )
            df = df[[term, "Cited by"]].head(top_n)
        df = df.reset_index(drop=True)
        plot = plots[plot_type]
        output.clear_output()
        with output:
            display(plot(df, cmap=cmap, figsize=FIGSIZE))

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

##
##
##  Panel 1
##
##
def __body_1(x):
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
                    options=["Countries", "Country 1st"],
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
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
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
        term = kwargs['term']
        analysis_type = kwargs['analysis_type']
        cmap = kwargs['cmap']
        #
        df = summary_by_term(x, term)
        if analysis_type == "Frequency":
            df = df[[term, "Num Documents"]]
        else:
            df = df[[term, "Cited by"]]
        df = df.reset_index(drop=True)
        output.clear_output()
        with output:
            display(plt.worldmap(df, figsize=FIGSIZE, cmap=cmap))        
        
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
    body.children = [__body_0(df), __body_1(df)]
    body.set_title(0, "Term Analysis")
    body.set_title(1, "Worldmap")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Summary by Term"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )

