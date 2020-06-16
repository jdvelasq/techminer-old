
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




# def get_top_terms(x, column, top_by=0, top_n=None, limit_to=None, exclude=None):
    
#     result = summary_by_term(x, column)

#     if top_by == 0 or top_by == "Frequency":
        
#     if top_by == 1 or top_by == "Times_Cited":






# def most_frequent(x, column, top_n, limit_to, exclude):
#     result = summary_by_term(x, column, limit_to=limit_to, exclude=exclude)
#     result = result.sort_values(["Num_Documents", "Times_Cited", column], ascending=False)
#     result = result[column].head(top_n)
#     result = result.tolist()
#     return result


# def most_cited_by(x, column, top_n, limit_to, exclude):
#     result = summary_by_term(x, column, limit_to=limit_to, exclude=exclude)
#     result = result.sort_values(["Times_Cited", "Num_Documents", column], ascending=False)
#     result = result[column].head(top_n)
#     result = result.tolist()
#     return result


def summary_by_term(x, column, top_by=None, top_n=None, limit_to=None, exclude=None):
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
    ...          "Times_Cited": list(range(10,14)),
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Times_Cited  ID
    0  author 0;author 1;author 2           10   0
    1                    author 0           11   1
    2                    author 1           12   2
    3                    author 3           13   3

    >>> summary_by_term(x, 'Authors')
        Authors  Num_Documents  Times_Cited      ID
    0  author 0              2           21  [0, 1]
    1  author 1              2           22  [0, 2]
    2  author 2              1           10     [0]
    3  author 3              1           13     [3]

    >>> items = ['author 1', 'author 2']
    >>> summary_by_term(x, 'Authors', limit_to=items)
        Authors  Num_Documents  Times_Cited      ID
    0  author 1              2           22  [0, 2]
    1  author 2              1           10     [0]

    >>> summary_by_term(x, 'Authors', exclude=items)
        Authors  Num_Documents  Times_Cited      ID
    0  author 0              2           21  [0, 1]
    1  author 3              1           13     [3]

    """

    #
    # Computation
    #

    x = x.copy()
    x = __explode(x[[column, "Times_Cited", "ID"]], column)
    x["Num_Documents"] = 1
    result = x.groupby(column, as_index=False).agg(
        {"Num_Documents": np.size, "Times_Cited": np.sum}
    )
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

    #
    # Filter
    #
    
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

    if (top_by == 0 or top_by == "Frequency"):
        result = result[[column, 'Num_Documents', "Times_Cited", 'ID']]
        result.sort_values(
            ["Num_Documents", "Times_Cited", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 1 or top_by == "Times_Cited"):
        result = result[[column, "Times_Cited", 'Num_Documents', 'ID']]
        result.sort_values(
            ["Times_Cited", "Num_Documents", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if top_by is None:
        result.sort_values(
            [column, "Num_Documents", "Times_Cited"],
            ascending=[True, False, False],
            inplace=True,
            ignore_index=True,
        )

    if top_n is not None:
        result = result.head(top_n)

    return result

def get_top_by(x, column, top_by, top_n, limit_to, exclude):
    """Return a list with the top_n terms of column
    """
    return summary_by_term(x=x, column=column, top_by=top_by, top_n=top_n, limit_to=limit_to, exclude=exclude)[column].tolist()




# def documents_by_term(x, column, limit_to=None, exclude=None):
#     """Computes the number of documents per term in a given column.

#     Args:
#         x (pandas.DataFrame): Bibliographic dataframe
#         column (str): Column to analize.
#         limit_to (list): Limits to the terms in the list from the results.
#         exclude (list): Terms to be excluded.
        

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...     {
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
#     ...          "Times_Cited": list(range(10,14)),
#     ...          "ID": list(range(4)),
#     ...     }
#     ... )
#     >>> x
#                           Authors  Times_Cited  ID
#     0  author 0;author 1;author 2        10   0
#     1                    author 0        11   1
#     2                    author 1        12   2
#     3                    author 3        13   3

#     >>> documents_by_term(x, 'Authors')
#         Authors  Num_Documents      ID
#     0  author 0              2  [0, 1]
#     1  author 1              2  [0, 2]
#     2  author 2              1     [0]
#     3  author 3              1     [3]

#     >>> terms = ['author 1', 'author 2']
#     >>> documents_by_term(x, 'Authors', limit_to=terms)
#         Authors  Num_Documents      ID
#     0  author 1              2  [0, 2]
#     1  author 2              1     [0]

#     """

#     result = summary_by_term(x, column, limit_to, exclude)
#     result.pop("Times_Cited")
#     result.sort_values(
#         ["Num_Documents", column],
#         ascending=[False, True],
#         inplace=True,
#         ignore_index=True,
#     )
#     return result


# def citations_by_term(x, column, limit_to=None, exclude=None):
#     """Computes the number of citations by item in a column.

#     Args:
#         x (pandas.DataFrame): bibliographic dataframe.
#         column (str): the column to analyze.
#         limit_to (list): Limits to the terms in the list from the results.
#         exclude (list): Terms to be excluded.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...     {
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3".split(","),
#     ...          "Times_Cited": list(range(10,14)),
#     ...          "ID": list(range(4)),
#     ...     }
#     ... )
#     >>> x
#                           Authors  Times_Cited  ID
#     0  author 0;author 1;author 2        10   0
#     1                    author 0        11   1
#     2                    author 1        12   2
#     3                    author 3        13   3

#     >>> citations_by_term(x, 'Authors')
#         Authors  Times_Cited      ID
#     0  author 1        22  [0, 2]
#     1  author 0        21  [0, 1]
#     2  author 3        13     [3]
#     3  author 2        10     [0]

#     >>> terms = ['author 1', 'author 2']
#     >>> citations_by_term(x, 'Authors', limit_to=terms)
#         Authors  Times_Cited      ID
#     0  author 1        22  [0, 2]
#     1  author 2        10     [0]


#     """
#     result = summary_by_term(x, column, limit_to, exclude)
#     result.pop("Num_Documents")
#     result.sort_values(
#         ["Times_Cited", column], ascending=[False, True], inplace=True, ignore_index=True,
#     )
#     return result



# def most_cited_documents(x):
#     """ Returns the most cited documents.

#     Args:
#         x (pandas.DataFrame): bibliographic dataframe.

#     Results:
#         A pandas.DataFrame.
        

#     """
#     result = x.sort_values(by="Times_Cited", ascending=False)[
#         ["Title", "Authors", "Year", "Times_Cited", "ID"]
#     ]
#     result["Times_Cited"] = result["Times_Cited"].map(
#         lambda w: int(w) if pd.isna(w) is False else 0
#     )
#     return result







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
#         ...       "Times_Cited": list(range(len(x))),
#         ...       'ID': list(range(len(x))),
#         ...    }
#         ... )
#         >>> df
#           Authors Author Keywords  Times_Cited   ID
#         0       A               a         0   0
#         1     A;B             a;b         1   1
#         2       B               b         2   2
#         3   A;B;C               c         3   3
#         4     B;D             c;d         4   4
#         5     A;B               d         5   5

#         >>> DataFrame(df).most_frequent('Authors', top_n=1)
#           Authors Author Keywords  Times_Cited   ID  top_1_Authors_freq
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
#         """Creates a group for most items Times_Cited 

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
#         >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
#         >>> df = pd.DataFrame(
#         ...    {
#         ...       'Authors': x,
#         ...       'Author Keywords': y,
#         ...       "Times_Cited": list(range(len(x))),
#         ...       'ID': list(range(len(x))),
#         ...    }
#         ... )
#         >>> df
#           Authors Author Keywords  Times_Cited   ID
#         0       A               a         0   0
#         1     A;B             a;b         1   1
#         2       B               b         2   2
#         3   A;B;C               c         3   3
#         4     B;D             c;d         4   4
#         5     A;B               d         5   5

#         >>> DataFrame(df).most_cited_by('Authors', top_n=1)
#           Authors Author Keywords  Times_Cited   ID  top_1_Authors_cited_by
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
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                    options=["Summary", "Bar plot", "Horizontal bar plot", "Pie plot", "Wordcloud", "Treemap"],
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                    options=[z for z in COLUMNS if z in x.columns],
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 2
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                    options=["Frequency", "Times_Cited"],
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 3
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                    options=list(range(5, 51, 5)),
                    ensure_option=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 4
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
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
        output.clear_output()
        with output:
            display(widgets.HTML('Processing ...'))
        #
        view = kwargs['view']
        column = kwargs['column']
        top_by = kwargs['top_by']
        top_n = kwargs['top_n']
        cmap = kwargs['cmap']
        figsize_width = int(kwargs['figsize_width'])
        figsize_height = int(kwargs['figsize_height'])
        #
        plots = {
            "Summary": None,
            "Bar plot": plt.bar_prop,
            "Horizontal bar plot": plt.barh_prop,
            "Pie plot": plt.pie_prop,
            "Wordcloud": plt.wordcloud,
            "Treemap": plt.tree,
        }
        #
        if view == 'Summary':
            controls[4]["widget"].disabled = True
            controls[-1]["widget"].disabled = True
            controls[-2]["widget"].disabled = True
        else:
            controls[4]["widget"].disabled = False
            controls[-1]["widget"].disabled = False
            controls[-2]["widget"].disabled = False
        #   
        df = summary_by_term(x, column=column, top_by=top_by, top_n = top_n, limit_to=limit_to, exclude=exclude)
        #
        plot = plots[view]
        output.clear_output()
        with output:
            if plot is None:
                df.pop('ID')
                display(df)
            else:
                if top_by == "Frequency":
                    df = df[[column, "Num_Documents", "Times_Cited"]]
                else:
                    df = df[[column, "Times_Cited", "Num_Documents" ]]

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
            df = df[[term, "Times_Cited"]]
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
    body.children = [__APP0__(df, limit_to, exclude), __APP1__(df, limit_to, exclude)]
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
