
"""
Analysis by Term
==================================================================================================



"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.explode import __explode
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS


def summary_by_term(x, column, limit_to=None, exclude=None):
    """Summarize the number of documents and citations by term in a dataframe.

    Args:
        x (pandas.DataFrame): Bibliographic dataframe
        column (str): Column to Analyze.
        limit_to (list): Limit the result to the terms in the list.
        exclude (list): Terms to be excluded.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited_by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited_by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> summary_by_term(x, 'Authors')
        Authors  Num_Documents  Cited_by      ID
    0  author 0              2        21  [0, 1]
    1  author 1              2        22  [0, 2]
    2  author 2              1        10     [0]
    3  author 3              1        13     [3]

    >>> items = ['author 1', 'author 2']
    >>> summary_by_term(x, 'Authors', limit_to=items)
        Authors  Num_Documents  Cited_by      ID
    0  author 1              2        22  [0, 2]
    1  author 2              1        10     [0]

    >>> summary_by_term(x, 'Authors', exclude=items)
        Authors  Num_Documents  Cited_by      ID
    0  author 0              2        21  [0, 1]
    1  author 3              1        13     [3]

    """
    x = x.copy()
    x = __explode(x[[column, "Cited_by", "ID"]], column)
    x["Num_Documents"] = 1
    result = x.groupby(column, as_index=False).agg(
        {"Num_Documents": np.size, "Cited_by": np.sum}
    )
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result["Cited_by"] = result["Cited_by"].map(lambda x: int(x))

    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        result = result[result[column].map(lambda w: w in limit_to)]

    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        result = result[result[column].map(lambda w: w not in exclude)]

    result.sort_values(
        [column, "Num_Documents", "Cited_by"],
        ascending=[True, False, False],
        inplace=True,
        ignore_index=True,
    )
    return result


def documents_by_term(x, column, limit_to=None, exclude=None):
    """Computes the number of documents per term in a given column.

    Args:
        x (pandas.DataFrame): Bibliographic dataframe
        column (str): Column to analize.
        limit_to (list): Limits to the terms in the list from the results.
        exclude (list): Terms to be excluded.
        

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited_by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited_by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> documents_by_term(x, 'Authors')
        Authors  Num_Documents      ID
    0  author 0              2  [0, 1]
    1  author 1              2  [0, 2]
    2  author 2              1     [0]
    3  author 3              1     [3]

    >>> terms = ['author 1', 'author 2']
    >>> documents_by_term(x, 'Authors', limit_to=terms)
        Authors  Num_Documents      ID
    0  author 1              2  [0, 2]
    1  author 2              1     [0]

    """

    result = summary_by_term(x, column, limit_to, exclude)
    result.pop("Cited_by")
    result.sort_values(
        ["Num_Documents", column],
        ascending=[False, True],
        inplace=True,
        ignore_index=True,
    )
    return result


def citations_by_term(x, column, limit_to=None, exclude=None):
    """Computes the number of citations by item in a column.

    Args:
        x (pandas.DataFrame): bibliographic dataframe.
        column (str): the column to analyze.
        limit_to (list): Limits to the terms in the list from the results.
        exclude (list): Terms to be excluded.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...     {
    ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
    ...          "Cited_by": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Cited_by  ID
    0  author 0;author 1;author 2        10   0
    1                    author 0        11   1
    2                    author 1        12   2
    3                    author 3        13   3

    >>> citations_by_term(x, 'Authors')
        Authors  Cited_by      ID
    0  author 1        22  [0, 2]
    1  author 0        21  [0, 1]
    2  author 3        13     [3]
    3  author 2        10     [0]

    >>> terms = ['author 1', 'author 2']
    >>> citations_by_term(x, 'Authors', limit_to=terms)
        Authors  Cited_by      ID
    0  author 1        22  [0, 2]
    1  author 2        10     [0]


    """
    result = summary_by_term(x, column, limit_to, exclude)
    result.pop("Num_Documents")
    result.sort_values(
        ["Cited_by", column], ascending=[False, True], inplace=True, ignore_index=True,
    )
    return result



def most_cited_documents(x):
    """ Returns the most cited documents.

    Args:
        x (pandas.DataFrame): bibliographic dataframe.

    Results:
        A pandas.DataFrame.
        

    """
    result = x.sort_values(by="Cited_by", ascending=False)[
        ["Title", "Authors", "Year", "Cited_by", "ID"]
    ]
    result["Cited_by"] = result["Cited_by"].map(
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


###############################################################################
##
##  APP
##
###############################################################################

WIDGET_WIDTH = "180px"
LEFT_PANEL_HEIGHT = "655px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "720px", 0]


##
##
##  Panel 0
##
##
def __APP0__(x, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
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
            "desc": "View:",
            "widget": widgets.Dropdown(
                    options=["Bar plot", "Horizontal bar plot", "Pie plot", "Wordcloud", "Treemap", "Table"],
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
            "widget": widgets.Dropdown(
                    options=list(range(5, 51, 5)),
                    ensure_option=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 5
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
                ),
        },
        # 6
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
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
        figsize_width = int(kwargs['figsize_width'])
        figsize_height = int(kwargs['figsize_height'])
        #
        plots = {
            "Bar plot": plt.bar,
            "Horizontal bar plot": plt.barh,
            "Pie plot": plt.pie,
            "Wordcloud": plt.wordcloud,
            "Treemap": plt.tree,
            "Table": None
        }
        #
        if plot_type == 'Table':
            controls[3]["widget"].disabled = True
            controls[-1]["widget"].disabled = True
            controls[-2]["widget"].disabled = True
        else:
            controls[3]["widget"].disabled = False
            controls[-1]["widget"].disabled = False
            controls[-2]["widget"].disabled = False
        #   
        df = summary_by_term(x, term, limit_to=limit_to, exclude=exclude)
        #
        if analysis_type == "Frequency":
            df = df.sort_values(
                ["Num_Documents", "Cited_by", term], ascending=False
            )
            df = df[[term, "Num_Documents"]].head(top_n)
        else:
            df = df.sort_values(
                ["Cited_by", "Num_Documents", term], ascending=False
            )
            df = df[[term, "Cited_by"]].head(top_n)
        df = df.reset_index(drop=True)
        plot = plots[plot_type]
        output.clear_output()
        with output:
            if plot is None:
                display(df)
            else:
                display(plot(df, cmap=cmap, figsize=(figsize_width, figsize_height)))
            
    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args))
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                    if control["desc"] not in ["Figsize"]
                ] + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([
                        controls[-2]["widget"],
                        controls[-1]["widget"],
                    ])
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
        ]
    )

#
#
#  Panel 1
#
#
def __APP1__(x, limit_to, exclude):
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
                    options=["Countries", "Country_1st_Author"],
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
        # 3
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(15, 21, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
                ),
        },
        # 4
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(4, 9, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
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
        figsize_width = int(kwargs['figsize_width'])
        figsize_height = int(kwargs['figsize_height'])
        #
        df = summary_by_term(x, term)
        if analysis_type == "Frequency":
            df = df[[term, "Num_Documents"]]
        else:
            df = df[[term, "Cited_by"]]
        df = df.reset_index(drop=True)
        output.clear_output()
        with output:
            display(plt.worldmap(df, figsize=(figsize_width, figsize_height), cmap=cmap))        
        
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
                    if control["desc"] not in ["Figsize"]
                ] + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([
                        controls[-2]["widget"],
                        controls[-1]["widget"],
                    ])
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox([output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")),
        ]
    )



def app(df, limit_to=None, exclude=None):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__APP0__(df, limit_to, exclude), __APP1__(df)]
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
