
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
    last_year = x.Year.max()

    x["SD"] = x[column].map(lambda w: 1 if isinstance(w, str) and len(w.split(';')) == 1 else 0)
    x["MD"] = x[column].map(lambda w: 1 if isinstance(w, str) and len(w.split(';')) > 1 else 0)

    x["Num_Documents"] = 1
    x['Frac_Num_Documents'] = x[column].map(lambda w: round(1 / len(w.split(';')), 2) if not pd.isna(w) else 0)
    x["First_Year"] = x.Year
    x = __explode(x[[column, "Num_Documents", "Times_Cited", "Frac_Num_Documents", "First_Year", "SD", "MD", "ID"]], column)
    result = x.groupby(column, as_index=False).agg(
        {
            "Num_Documents": np.sum, 
            "Times_Cited": np.sum, 
            "Frac_Num_Documents": np.sum, 
            "First_Year": np.min, 
            "SD": np.sum,
            "MD": np.sum,
        }
    )
    result['Last_Year'] = last_year
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result = result.assign(Years = result.Last_Year - result.First_Year + 1)
    result = result.assign(Times_Cited_per_Year = result.Times_Cited / result.Years)
    result["Times_Cited_per_Year"] = result["Times_Cited_per_Year"].map(lambda w: round(w, 2))
    result = result.assign(Avg_Times_Cited = result.Times_Cited / result.Num_Documents)
    result["Avg_Times_Cited"] = result["Avg_Times_Cited"].map(lambda w: round(w, 2))
    
    result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

    result["SMR"] = [ round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)]

    

    #
    # Indice H
    #
    z = x[[column, 'Times_Cited', "ID"]].copy()
    z = (
        x.assign(
            rn=x.sort_values('Times_Cited', ascending=False).groupby(column).cumcount() + 1
        )
    ).sort_values([column, 'Times_Cited', 'rn'], ascending=[False,False,True])
    z['rn2'] = z.rn.map(lambda w: w * w)

    q = z.query('Times_Cited >= rn')
    q = q.groupby(column, as_index=False).agg({'rn': np.max})
    h_dict = {key: value for key, value in zip(q[column], q.rn)}

    result['H_index'] = result[column].map(lambda w: h_dict[w] if w in h_dict.keys() else 0)

    #
    # indice M
    #
    result = result.assign(M_index=result.H_index / result.Years)
    result["M_index"] = result["M_index"].map(lambda w: round(w, 2))

    #
    # indice G
    #
    q = z.query('Times_Cited >= rn2')
    q = q.groupby(column, as_index=False).agg({'rn': np.max})
    h_dict = {key: value for key, value in zip(q[column], q.rn)}
    result['G_index'] = result[column].map(lambda w: h_dict[w] if w in h_dict.keys() else 0)


    #
    # Orden de las columnas
    #
    result = result[
        [
            column, 
            "Num_Documents", 
            "Frac_Num_Documents", 
            "Times_Cited",  
            "Times_Cited_per_Year", 
            "Avg_Times_Cited",
            "H_index",
            "M_index",
            "G_index",
            "SD",
            "MD",
            "SMR",
            "First_Year", 
            "Last_Year", 
            "Years",
            "ID",
        ]
    ]

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

    if (top_by == 0 or top_by == "Num Documents"):
        result.sort_values(
            ["Num_Documents", "Times_Cited", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 1 or top_by == "Times Cited"):
        result.sort_values(
            ["Times_Cited", "Frac_Num_Documents", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 2 or top_by == "Frac Num Documents"):
        result.sort_values(
            ["Frac_Num_Documents", "Times_Cited", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 3 or top_by == "Times Cited per Year"):
        result.sort_values(
            ["Times_Cited_per_Year", "Frac_Num_Documents", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 4 or top_by == "Avg Times Cited"):
        result.sort_values(
            ["Avg_Times_Cited", "Frac_Num_Documents", column],
            ascending=[False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 5 or top_by == "H index"):
        result.sort_values(
            ["H_index", "G_index", "Times_Cited", column],
            ascending=[False, False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 6 or top_by == "M index"):
        result.sort_values(
            ["M_index", "G_index", "Times_Cited", column],
            ascending=[False, False, False, True],
            inplace=True,
            ignore_index=True,
        )

    if (top_by == 7 or top_by == "G index"):
        result.sort_values(
            ["G_index", "H_index", "Times_Cited", column],
            ascending=[False, False, False, True],
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
                    options=[
                        "Summary",
                        "Bar plot",
                        "Horizontal bar plot",
                        "Pie plot",
                        "Wordcloud",
                        "Treemap",
                        "S/D Ratio (bar)",
                        "S/D Ratio (barh)",
                    ],
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
                    options=[
                        "Num Documents", 
                        "Times Cited",
                        "Frac Num Documents",
                        "Times Cited per Year",
                        "Avg Times Cited",
                        "H index",
                        "M index",
                        "G index",
                    ],
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
            "arg": "sort_by",
            "desc": "Sort by:",
            "widget": widgets.Dropdown(
                    options=[
                        "Num Documents asc",
                        "Num Documents desc",
                        "Frac Num Documents asc",
                        "Frac Num Documents desc",
                        "Times Cited asc",
                        "Times Cited desc", 
                        "Times Cited per Year asc",
                        "Times Cited per Year desc",
                        "Avg Times Cited asc",
                        "Avg Times Cited desc",
                        "H index asc",
                        "H index desc",
                        "M index asc",
                        "M index desc",
                        "G index asc",
                        "G index desc",
                        "Column asc",
                        "Column desc",
                    ],
                    value="Num Documents desc",
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 5
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 6
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                    options=range(5,15, 1),
                    ensure_option=True,
                    layout=Layout(width="88px"),
                ),
        },
        # 7
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
        sort_by = kwargs['sort_by']
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
            "S/D Ratio (bar)": None,
            "S/D Ratio (barh)": None,
        }
        #
        if view == 'Summary':
            controls[5]["widget"].disabled = True
            controls[-1]["widget"].disabled = True
            controls[-2]["widget"].disabled = True
        else:
            controls[5]["widget"].disabled = False
            controls[-1]["widget"].disabled = False
            controls[-2]["widget"].disabled = False
        #   
        df = summary_by_term(x, column=column, top_by=top_by, top_n = top_n, limit_to=limit_to, exclude=exclude)
        #
        if sort_by == "Num Documents asc":
            df = df.sort_values(by=["Num_Documents", "Times_Cited", df.columns[0]], ascending=True)
        if sort_by == "Num Documents desc":
            df = df.sort_values(by=["Num_Documents", "Times_Cited", df.columns[0]], ascending=False)
        if sort_by == "Frac Num Documents asc":
            df = df.sort_values(by=["Frac_Num_Documents", "Times_Cited", df.columns[0]], ascending=True)
        if sort_by == "Frac Num Documents desc":
            df = df.sort_values(by=["Frac_Num_Documents", "Times_Cited", df.columns[0]], ascending=False)
        if sort_by == "Times Cited asc":
            df = df.sort_values(by=["Times_Cited", "Num_Documents", df.columns[0]], ascending=True)
        if sort_by == "Times Cited desc":
            df = df.sort_values(by=["Times_Cited", "Num_Documents", df.columns[0]], ascending=False)
        if sort_by == "Times Cited per Year asc":
            df = df.sort_values(by=["Times_Cited_per_Year", "Num_Documents", df.columns[0]], ascending=True)
        if sort_by == "Times Cited per Year desc":
            df = df.sort_values(by=["Times_Cited_per_Year", "Num_Documents", df.columns[0]], ascending=False)
        if sort_by == "Avg Times Cited asc":
            df = df.sort_values(by=["Avg_Times_Cited", "Num_Documents", df.columns[0]], ascending=True)
        if sort_by == "Avg Times Cited desc":
            df = df.sort_values(by=["Avg_Times_Cited", "Num_Documents", df.columns[0]], ascending=False)
        if sort_by == "Column asc":
            df = df.sort_values(by=[df.columns[0], "Times_Cited", "Num_Documents"], ascending=True)
        if sort_by == "Column desc":
            df = df.sort_values(by=[df.columns[0], "Times_Cited", "Num_Documents"], ascending=False)

        if sort_by == "H index asc":
            df = df.sort_values(by=["H_index", "G_index", "Times_Cited", "Num_Documents"], ascending=True)
        if sort_by == "H index desc":
            df = df.sort_values(by=["H_index", "G_index", "Times_Cited", "Num_Documents"], ascending=False)
        if sort_by == "M index asc":
            df = df.sort_values(by=["M_index", "G_index", "Times_Cited", "Num_Documents"], ascending=True)
        if sort_by == "M index desc":
            df = df.sort_values(by=["M_index", "G_index", "Times_Cited", "Num_Documents"], ascending=False)
        if sort_by == "G index asc":
            df = df.sort_values(by=["G_index", "H_index", "Times_Cited", "Num_Documents"], ascending=True)
        if sort_by == "G index desc":
            df = df.sort_values(by=["G_index", "H_index", "Times_Cited", "Num_Documents"], ascending=False)

        #
        df = df.reset_index(drop=True)
        #
        plot = plots[view]
        output.clear_output()
        with output:
            if view == "Summary":
                df.pop('ID')
                display(df)
            else:

                if view == "S/D Ratio (bar)":
                    display(plt.stacked_bar(df[[column, "SD", "MD"]], figsize=(figsize_width, figsize_height), cmap=cmap))
                    return

                if view == "S/D Ratio (barh)":
                    display(plt.stacked_barh(df[[column, "SD", "MD"]], figsize=(figsize_width, figsize_height), cmap=cmap))
                    return

                if top_by == "Num Documents":
                    df = df[[column, "Num_Documents", "Times_Cited"]]
                if top_by == "Times Cited":
                    df = df[[column, "Times_Cited", "Num_Documents"]]
                if top_by == "Frac Num Documents":
                    df = df[[column, "Frac_Num_Documents", "Times_Cited"]]
                if top_by == "Times Cited per Year":
                    df = df[[column, "Times_Cited_per_Year", "Num_Documents"]]
                if top_by == "Avg Times Cited":
                    df = df[[column, "Avg_Times_Cited", "Num_Documents"]]
                if top_by == "H index":
                    df = df[[column, "H_index", "Avg_Times_Cited"]]
                if top_by == "M index":
                    df = df[[column, "M_index", "Avg_Times_Cited"]]
                if top_by == "G index":
                    df = df[[column, "G_index", "Avg_Times_Cited"]]

    
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
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                    options=["Countries", "Country_1st_Author"],
                    layout=Layout(width=WIDGET_WIDTH),
                ),
        },
        # 1
        {
            "arg": "analysis_type",
            "desc": "Analysis type:",
            "widget": widgets.Dropdown(
                    options=["Num Documents", "Times Cited"],
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
        if analysis_type == "Num Documents":
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
    body.children = [__APP0__(df, limit_to, exclude), __APP1__(df, limit_to, exclude), ]
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
