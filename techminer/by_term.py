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


#
#
#  Top documents
#
#
def top_documents(data):
    """Returns the top 50 documents by Times Cited.

    Args:
        data (pandas.DataFrame): A bibliographic dataframe.

    Returns:
    """
    data = data.sort_values(["Times_Cited", "Year"], ascending=[False, True])
    data = data.head(50)
    data["Times_Cited"] = data.Times_Cited.map(lambda w: int(w))
    data = data.reset_index(drop=True)
    return data


def __APP4__(data):
    """
    # >>> import pandas as pd
    # >>> data = pd.DataFrame(
    # ...     {
    # ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    # ...          "Times_Cited": list(range(10,16)),
    # ...          "ID": list(range(6)),
    # ...     }
    # ... )
    # >>> __APP4__(data)


    """
    result = data.sort_values(["Times_Cited", "Title"], ascending=[False, True])
    result = result[["Authors", "Year", "Title", "Source_title", "Times_Cited"]]
    result = result.head(50)
    output = widgets.Output()
    with output:
        display(result)

    return widgets.HBox([output])


#
#
#  Core source titles
#
#


def core_source_titles(data):
    """[summary]

    Args:
        data ([type]): [description]
    """
    m = summary(
        data, "Source_title", top_by=None, top_n=None, limit_to=None, exclude=None
    )
    m = m[["Num_Documents"]]
    m = m.groupby(["Num_Documents"]).size()
    w = [str(round(100 * a / sum(m), 2)) + " %" for a in m]
    m = pd.DataFrame(
        {"Num Sources": m.tolist(), "%": w, "Documents published": m.index}
    )

    m = m.sort_values(["Documents published"], ascending=False)
    m["Acum Num Sources"] = m["Num Sources"].cumsum()
    m["% Acum"] = [
        str(round(100 * a / sum(m["Num Sources"]), 2)) + " %"
        for a in m["Acum Num Sources"]
    ]

    m["Tot Documents published"] = m["Num Sources"] * m["Documents published"]
    m["Num Documents"] = m["Tot Documents published"].cumsum()
    m["Tot Documents"] = m["Num Documents"].map(
        lambda w: str(round(w / m["Num Documents"].max() * 100, 2)) + " %"
    )

    bradford1 = int(len(data) / 3)
    bradford2 = 2 * bradford1

    m["Bradford's Group"] = m["Num Documents"].map(
        lambda w: 3 if w > bradford2 else (2 if w > bradford1 else 1)
    )

    m = m[
        [
            "Num Sources",
            "%",
            "Acum Num Sources",
            "% Acum",
            "Documents published",
            "Tot Documents published",
            "Num Documents",
            "Tot Documents",
            "Bradford's Group",
        ]
    ]

    m = m.reset_index(drop=True)
    return m


def __APP3__(data):
    output = widgets.Output()
    with output:
        display(core_source_titles(data))
    return widgets.HBox([output])


#
#
#  Core Authors
#
#
def core_authors(data):
    """
    """
    #
    # Numero de documentos escritos por author
    #
    z = summary(data, "Authors", top_by=None, top_n=None, limit_to=None, exclude=None)

    authors_dict = {
        author: num_docs
        for author, num_docs in zip(z.Authors, z.Num_Documents)
        if not pd.isna(author)
    }

    z = z[["Num_Documents"]]
    z = z.groupby(["Num_Documents"]).size()
    w = [str(round(100 * a / sum(z), 2)) + " %" for a in z]
    z = pd.DataFrame(
        {"Num Authors": z.tolist(), "%": w, "Documents written per Author": z.index}
    )
    z = z.sort_values(["Documents written per Author"], ascending=False)
    z["Acum Num Authors"] = z["Num Authors"].cumsum()
    z["% Acum"] = [
        str(round(100 * a / sum(z["Num Authors"]), 2)) + " %"
        for a in z["Acum Num Authors"]
    ]

    m = __explode(data[["Authors", "ID"]], "Authors")
    m = m.dropna()
    m["Documents_written"] = m.Authors.map(lambda w: authors_dict[w])
    n = []
    for k in z["Documents written per Author"]:
        s = m.query("Documents_written >= " + str(k))
        s = s[["ID"]]
        s = s.drop_duplicates()
        n.append(len(s))

    k = []
    for index in range(len(n) - 1):
        k.append(n[index + 1] - n[index])
    k = [n[0]] + k
    z["Num Documents"] = k
    z["Acum Num Documents"] = n

    z = z[
        [
            "Num Authors",
            "%",
            "Acum Num Authors",
            "% Acum",
            "Documents written per Author",
            "Num Documents",
            "Acum Num Documents",
        ]
    ]

    z = z.reset_index(drop=True)
    return z


#
#
#  Panel 2
#
#
def __APP2__(data):
    output = widgets.Output()
    with output:
        display(core_authors(data).head(50))
    return widgets.HBox([output])


#
#
#  worldmap
#
#
def worldmap(
    result, top_by="Num_Documents", cmap="Greys", figsize=(10, 5), fontsize=12,
):
    """
    """
    return plt.worldmap(
        result[[result.columns[0], top_by]],
        cmap=cmap,
        figsize=figsize,
        fontsize=fontsize,
    )


def __APP1__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=["Countries", "Country_1st_Author"],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num_Documents", "Times_Cited"],
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
            "arg": "width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(15, 21, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 4
        {
            "arg": "height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(4, 9, 1), ensure_option=True, layout=Layout(width="88px"),
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
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        cmap = kwargs["cmap"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        #
        result = summary(
            data, column=column, output=0, top_by="Num_Documents", top_n=None,
        )
        #
        output.clear_output()
        with output:
            display(
                worldmap(result, top_by=top_by, figsize=(width, height), cmap=cmap,)
            )

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
                ]
                + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([controls[-2]["widget"], controls[-1]["widget"],]),
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


def summary(
    data,
    column,
    output=0,
    top_by=None,
    top_n=None,
    sort_by="Num_Documents",
    ascending=True,
    cmap="Greys",
    limit_to=None,
    exclude=None,
    fontsize=12,
    figsize=(10, 5),
):
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
    ...          "Year": [1990, 1990, 1991, 1991],
    ...          "ID": list(range(4)),
    ...     }
    ... )
    >>> x
                          Authors  Times_Cited  Year  ID
    0  author 0;author 1;author 2           10  1990   0
    1                    author 0           11  1990   1
    2                    author 1           12  1991   2
    3                    author 3           13  1991   3

    >>> summary(x, 'Authors')
        Authors  Num_Documents  Times_Cited      ID
    0  author 0              2           21  [0, 1]
    1  author 1              2           22  [0, 2]
    2  author 2              1           10     [0]
    3  author 3              1           13     [3]

    >>> items = ['author 1', 'author 2']
    >>> summary(x, 'Authors', limit_to=items)
        Authors  Num_Documents  Times_Cited      ID
    0  author 1              2           22  [0, 2]
    1  author 2              1           10     [0]

    >>> summary(x, 'Authors', exclude=items)
        Authors  Num_Documents  Times_Cited      ID
    0  author 0              2           21  [0, 1]
    1  author 3              1           13     [3]

    """

    #
    # Computation
    #
    x = data.copy()
    last_year = x.Year.max()

    x["SD"] = x[column].map(
        lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
    )
    x["MD"] = x[column].map(
        lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
    )

    x["Num_Documents"] = 1
    x["Frac_Num_Documents"] = x[column].map(
        lambda w: round(1 / len(w.split(";")), 2) if not pd.isna(w) else 0
    )
    x["First_Year"] = x.Year
    x = __explode(
        x[
            [
                column,
                "Num_Documents",
                "Times_Cited",
                "Frac_Num_Documents",
                "First_Year",
                "SD",
                "MD",
                "ID",
            ]
        ],
        column,
    )
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
    result["Last_Year"] = last_year
    result = result.assign(ID=x.groupby(column).agg({"ID": list}).reset_index()["ID"])
    result = result.assign(Years=result.Last_Year - result.First_Year + 1)
    result = result.assign(Times_Cited_per_Year=result.Times_Cited / result.Years)
    result["Times_Cited_per_Year"] = result["Times_Cited_per_Year"].map(
        lambda w: round(w, 2)
    )
    result = result.assign(Avg_Times_Cited=result.Times_Cited / result.Num_Documents)
    result["Avg_Times_Cited"] = result["Avg_Times_Cited"].map(lambda w: round(w, 2))

    result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

    result["SMR"] = [round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)]

    #
    # Indice H
    #
    z = x[[column, "Times_Cited", "ID"]].copy()
    z = (
        x.assign(
            rn=x.sort_values("Times_Cited", ascending=False).groupby(column).cumcount()
            + 1
        )
    ).sort_values([column, "Times_Cited", "rn"], ascending=[False, False, True])
    z["rn2"] = z.rn.map(lambda w: w * w)

    q = z.query("Times_Cited >= rn")
    q = q.groupby(column, as_index=False).agg({"rn": np.max})
    h_dict = {key: value for key, value in zip(q[column], q.rn)}

    result["H_index"] = result[column].map(
        lambda w: h_dict[w] if w in h_dict.keys() else 0
    )

    #
    # indice M
    #
    result = result.assign(M_index=result.H_index / result.Years)
    result["M_index"] = result["M_index"].map(lambda w: round(w, 2))

    #
    # indice G
    #
    q = z.query("Times_Cited >= rn2")
    q = q.groupby(column, as_index=False).agg({"rn": np.max})
    h_dict = {key: value for key, value in zip(q[column], q.rn)}
    result["G_index"] = result[column].map(
        lambda w: h_dict[w] if w in h_dict.keys() else 0
    )

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

    #
    # Top by
    #
    if isinstance(top_by, str):
        top_by = top_by.replace(" ", "_")
        top_by = {
            "Num_Documents": 0,
            "Times_Cited": 1,
            "Frac_Num_Documents": 2,
            "Times_Cited_per_Year": 3,
            "Avg_Times_Cited": 4,
            "H_index": 5,
            "M_index": 6,
            "G_index": 7,
        }[top_by]

    if top_by is not None:

        by, ascending_top_by = {
            0: (["Num_Documents", "Times_Cited", column], [False, False, True]),
            1: (["Times_Cited", "Frac_Num_Documents", column], [False, False, True]),
            2: (["Frac_Num_Documents", "Times_Cited", column], [False, False, True]),
            3: (
                ["Times_Cited_per_Year", "Frac_Num_Documents", column],
                [False, False, True],
            ),
            4: (
                ["Avg_Times_Cited", "Frac_Num_Documents", column],
                [False, False, True],
            ),
            5: (
                ["H_index", "G_index", "Times_Cited", column],
                [False, False, False, True],
            ),
            6: (
                ["M_index", "G_index", "Times_Cited", column],
                [False, False, False, True],
            ),
            7: (
                ["G_index", "H_index", "Times_Cited", column],
                [False, False, False, True],
            ),
        }[top_by]

        result.sort_values(
            by=by, ascending=ascending_top_by, inplace=True, ignore_index=True,
        )

    else:

        result.sort_values(
            [column, "Num_Documents", "Times_Cited"],
            ascending=[True, False, False],
            inplace=True,
            ignore_index=True,
        )

    if top_n is not None:
        result = result.head(top_n)

    #
    #
    #  Sort by
    #
    #
    if isinstance(sort_by, str):
        sort_by = sort_by.replace(" ", "_")
        sort_by = {
            "Column": 0,
            "Num_Documents": 1,
            "Frac_Num_Documents": 2,
            "Times_Cited": 3,
            "Times_Cited_per_Year": 4,
            "Avg_Times_Cited": 5,
            "H_index": 6,
            "M_index": 7,
            "G_index": 8,
        }[sort_by]

    column0 = result.columns[0]
    sort_by = {
        0: [column0, "Times_Cited", "Num_Documents"],
        1: ["Num_Documents", "Times_Cited", column0],
        2: ["Frac_Num_Documents", "Times_Cited", column0],
        3: ["Times_Cited", "Num_Documents", column0],
        4: ["Times_Cited_per_Year", "Num_Documents", column0],
        5: ["Avg_Times_Cited", "Num_Documents", column0],
        6: ["H_index", "G_index", "Times_Cited", "Num_Documents"],
        7: ["M_index", "G_index", "Times_Cited", "Num_Documents"],
        8: ["G_index", "H_index", "Times_Cited", "Num_Documents"],
    }[sort_by]

    result = result.sort_values(by=sort_by, ascending=ascending)

    result = result.reset_index(drop=True)

    #
    #
    # Output
    #
    #
    if isinstance(output, str):
        output = output.replace(" ", "_")
        output = {
            "Summary": 0,
            "Bar_plot": 1,
            "Horizontal_bar_plot": 2,
            "Pie_plot": 3,
            "Wordcloud": 4,
            "Treemap": 5,
            "S/D_Ratio_(bar)": 6,
            "S/D_Ratio_(barh)": 7,
        }[output]

    if output == 0:
        result.pop("ID")
        return result

    values, prop_to = {
        0: ("Num_Documents", "Times_Cited"),
        1: ("Times_Cited", "Num_Documents"),
        2: ("Frac_Num_Documents", "Times_Cited"),
        3: ("Times_Cited_per_Year", "Num_Documents"),
        4: ("Avg_Times_Cited", "Num_Documents"),
        5: ("H_index", "Avg_Times_Cited"),
        6: ("M_index", "Avg_Times_Cited"),
        7: ("G_index", "Avg_Times_Cited"),
    }[top_by]

    if output in [1, 2, 3]:

        plot = {1: plt.bar, 2: plt.barh, 3: plt.pie,}[output]

        return plot(
            data=result,
            column=values,
            prop_to=prop_to,
            cmap=cmap,
            figsize=figsize,
            fontsize=fontsize,
        )

    if output in [4, 5]:
        plot = {4: plt.wordcloud, 5: plt.treemap,}[output]
        return plot(
            data=result, column=values, prop_to=prop_to, cmap=cmap, figsize=figsize
        )

    if output in [6, 7]:
        plot = {6: plt.stacked_bar, 7: plt.stacked_barh,}[output]
        return plot(
            result[[column, "SD", "MD"]],
            columns=["SD", "MD"],
            figsize=figsize,
            cmap=cmap,
        )

    return "ERROR: Output code unknown"


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
def __APP0__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
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
                options=[z for z in COLUMNS if z in data.columns],
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
                    "Num Documents",
                    "Frac Num Documents",
                    "Times Cited",
                    "Times Cited per Year",
                    "Avg Times Cited",
                    "H index",
                    "M index",
                    "G index",
                    "Column",
                ],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 5
        {
            "arg": "ascending",
            "desc": "Ascending:",
            "widget": widgets.Dropdown(
                options=["True", "False"], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 6
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 7
        {
            "arg": "width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
                ensure_option=True,
                layout=Layout(width="88px"),
            ),
        },
        # 8
        {
            "arg": "height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1),
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
            display(widgets.HTML("Processing ..."))
        #
        view = kwargs["view"]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = kwargs["top_n"]
        cmap = kwargs["cmap"]
        sort_by = kwargs["sort_by"]
        ascending = {"True": True, "False": False}[kwargs["ascending"]]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        if view == "Summary":
            controls[6]["widget"].disabled = True
            controls[-1]["widget"].disabled = True
            controls[-2]["widget"].disabled = True
        else:
            controls[6]["widget"].disabled = False
            controls[-1]["widget"].disabled = False
            controls[-2]["widget"].disabled = False

        output.clear_output()
        with output:
            display(
                summary(
                    data=data,
                    column=column,
                    output=view,
                    top_by=top_by,
                    top_n=top_n,
                    sort_by=sort_by,
                    ascending=ascending,
                    cmap=cmap,
                    limit_to=limit_to,
                    exclude=exclude,
                    figsize=(width, height),
                )
            )

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
                ]
                + [
                    widgets.Label(value="Figure Size"),
                    widgets.HBox([controls[-2]["widget"], controls[-1]["widget"],]),
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


#
#
#  Panel 1
#
#


def app(df, limit_to=None, exclude=None):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [
        __APP0__(df, limit_to, exclude),
        __APP1__(df, limit_to, exclude),
        __APP2__(df),
        __APP3__(df),
        __APP4__(df),
    ]
    body.set_title(0, "Term Analysis")
    body.set_title(1, "Worldmap")
    body.set_title(2, "Core Authors")
    body.set_title(3, "Core Source titles")
    body.set_title(4, "Top Documents")

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
