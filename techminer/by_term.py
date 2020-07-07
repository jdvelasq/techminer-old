"""
Analysis by Term
==================================================================================================



"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

import techminer.plots as plt
from techminer.explode import __explode
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS

import techminer.gui as gui


def analytics(
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
    fontsize=11,
    figsize=(6, 6),
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

    >>> analytics(x, 'Authors')[['Num_Documents', 'Times_Cited']]
              Num_Documents  Times_Cited
    Authors                             
    author 2              1           10
    author 3              1           13
    author 0              2           21
    author 1              2           22

    >>> items = ['author 1', 'author 2']
    >>> analytics(x, 'Authors', limit_to=items)[['Num_Documents', 'Times_Cited']]
              Num_Documents  Times_Cited
    Authors                             
    author 2              1           10
    author 1              2           22
    

    >>> analytics(x, 'Authors', exclude=items)[['Num_Documents', 'Times_Cited']]
              Num_Documents  Times_Cited
    Authors                             
    author 3              1           13
    author 0              2           21
    

    """

    #
    # Computation
    #
    x = data.copy()
    last_year = x.Year.max()

    # SD: single document / MD: multiple document
    x["SD"] = x[column].map(
        lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
    )
    x["MD"] = x[column].map(
        lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
    )

    x["Num_Documents"] = 1
    x["Frac_Num_Documents"] = x[column].map(
        lambda w: round(1 / (len(w.split(";")) if isinstance(w, str) else 1), 2)
        if not pd.isna(w)
        else 0
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
    result = result.reset_index(drop=True)
    result = result.set_index(column)

    result = result[
        [
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
    # Limit to
    #
    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        index = [w for w in result.index if w in limit_to]
        result = result.loc[index, :]

    #
    # Exclude
    #
    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        index = [w for w in result.index if w not in exclude]
        result = result.loc[index, :]

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
            0: (["Num_Documents", "Times_Cited"], False),
            1: (["Times_Cited", "Frac_Num_Documents"], False),
            2: (["Frac_Num_Documents", "Times_Cited"], False),
            3: (["Times_Cited_per_Year", "Frac_Num_Documents"], False,),
            4: (["Avg_Times_Cited", "Frac_Num_Documents"], False,),
            5: (["H_index", "G_index", "Times_Cited"], False,),
            6: (["M_index", "G_index", "Times_Cited"], False,),
            7: (["G_index", "H_index", "Times_Cited"], False,),
        }[top_by]

        result.sort_values(
            by=by, ascending=ascending_top_by, inplace=True,
        )

    else:

        result.sort_values(
            ["Num_Documents", "Times_Cited"], ascending=False, inplace=True,
        )

    if top_n is not None:
        result = result.head(top_n)

    #
    #  sort_by
    #
    if isinstance(sort_by, str):
        sort_by = sort_by.replace(" ", "_")
        sort_by = {
            "Num_Documents": 0,
            "Frac_Num_Documents": 1,
            "Times_Cited": 2,
            "Times_Cited_per_Year": 3,
            "Avg_Times_Cited": 4,
            "H_index": 5,
            "M_index": 6,
            "G_index": 7,
            "*Index*": 8,
        }[sort_by]

    if sort_by == 8:
        result = result.sort_index(axis=0, ascending=ascending)
    else:
        sort_by = {
            0: ["Num_Documents", "Times_Cited", "H_index"],
            1: ["Frac_Num_Documents", "Times_Cited", "H_index"],
            2: ["Times_Cited", "Num_Documents", "H_index"],
            3: ["Times_Cited_per_Year", "Num_Documents", "Times_Cited"],
            4: ["Avg_Times_Cited", "Num_Documents", "Times_Cited"],
            5: ["H_index", "G_index", "Times_Cited", "Num_Documents"],
            6: ["M_index", "G_index", "Times_Cited", "Num_Documents"],
            7: ["G_index", "H_index", "Times_Cited", "Num_Documents"],
        }[sort_by]
        result = result.sort_values(by=sort_by, ascending=ascending)

    #
    #
    # Output
    #
    #
    if isinstance(output, str):
        output = output.replace(" ", "_")
        output = {
            "Analytics": 0,
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
        #  if cmap is None:
        #      return result
        # return result.style.background_gradient(cmap=cmap, axis=0)

    values, darkness = {
        0: ("Num_Documents", "Times_Cited"),
        1: ("Times_Cited", "Num_Documents"),
        2: ("Frac_Num_Documents", "Times_Cited"),
        3: ("Times_Cited_per_Year", "Num_Documents"),
        4: ("Avg_Times_Cited", "Num_Documents"),
        5: ("H_index", "Avg_Times_Cited"),
        6: ("M_index", "Avg_Times_Cited"),
        7: ("G_index", "Avg_Times_Cited"),
    }[top_by]

    if output == 1:
        return plt.bar(
            height=result[values],
            darkness=result[darkness],
            cmap=cmap,
            figsize=figsize,
        )

    if output == 2:
        return plt.barh(
            width=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 3:
        return plt.pie(
            x=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 4:
        return plt.wordcloud(
            x=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 5:
        return plt.treemap(
            x=result[values], darkness=result[darkness], cmap=cmap, figsize=figsize,
        )

    if output == 6:
        z = result[["SD", "MD"]]
        return plt.stacked_bar(X=z, cmap=cmap, figsize=figsize,)

    if output == 7:
        z = result[["SD", "MD"]]
        return plt.stacked_barh(X=z, cmap=cmap, figsize=figsize,)

    return "ERROR: Output code unknown"


###############################################################################
##
##  APP
##
###############################################################################


def __TAB0__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        gui.dropdown(
            desc="View:",
            options=[
                "Analytics",
                "Bar plot",
                "Horizontal bar plot",
                "Pie plot",
                "Wordcloud",
                "Treemap",
                "S/D Ratio (bar)",
                "S/D Ratio (barh)",
            ],
        ),
        gui.dropdown(
            desc="Column:", options=[z for z in COLUMNS if z in data.columns],
        ),
        gui.dropdown(
            desc="Top by:",
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
        ),
        gui.top_n(),
        gui.dropdown(
            desc="Sort by:",
            options=[
                "Num Documents",
                "Frac Num Documents",
                "Times Cited",
                "Times Cited per Year",
                "Avg Times Cited",
                "H index",
                "M index",
                "G index",
                "*Index*",
            ],
        ),
        gui.ascending(),
        gui.cmap(),
        gui.fig_width(),
        gui.fig_height(),
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
        ascending = kwargs["ascending"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        if view == "Analytics":
            left_panel[-1]["widget"].disabled = True
            left_panel[-2]["widget"].disabled = True
        else:
            left_panel[-1]["widget"].disabled = False
            left_panel[-2]["widget"].disabled = False

        out = analytics(
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

        output.clear_output()
        with output:
            if view == "Analytics":
                display(out.style.background_gradient(cmap=cmap, axis=0))
            else:
                display(out)

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    # args = {control["arg"]: control["widget"] for control in left_panel}
    # output = widgets.Output()
    # with output:
    #     display(widgets.interactive_output(server, args))

    # grid = GridspecLayout(13, 6, height="650px")
    # #
    # # Left panel
    # #
    # for index in range(len(left_panel)):
    #     grid[index, 0] = widgets.HBox(
    #         [
    #             widgets.Label(value=left_panel[index]["desc"]),
    #             left_panel[index]["widget"],
    #         ],
    #         layout=Layout(
    #             display="flex", justify_content="flex-end", align_content="center",
    #         ),
    #     )
    # #
    # # Output
    # #
    # grid[0:, 1:] = widgets.VBox(
    #     [output], layout=Layout(height="650px", border="2px solid gray")
    # )

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


#
#
#
#
def __TAB1__(data):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    left_panel = [
        gui.dropdown(desc="Column:", options=["Countries", "Country_1st_Author"],),
        gui.dropdown(desc="Top by:", options=["Num_Documents", "Times_Cited"],),
        gui.cmap(),
        gui.fig_width(),
        gui.fig_height(),
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
        result = analytics(
            data, column=column, output=0, top_by="Num_Documents", top_n=None,
        )
        #
        output.clear_output()
        with output:
            display(plt.worldmap(x=result[top_by], figsize=(width, height), cmap=cmap,))

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    # args = {control["arg"]: control["widget"] for control in left_panel}
    # output = widgets.Output()
    # with output:
    #     display(widgets.interactive_output(server, args,))
    # #
    # grid = GridspecLayout(13, 6)
    # #
    # # Left panel
    # #
    # for index in range(len(left_panel)):
    #     grid[index, 0] = widgets.HBox(
    #         [
    #             widgets.Label(value=left_panel[index]["desc"]),
    #             left_panel[index]["widget"],
    #         ],
    #         layout=Layout(
    #             display="flex", justify_content="flex-end", align_content="center",
    #         ),
    #     )
    # #
    # # Output
    # #
    # grid[0:, 1:] = widgets.VBox(
    #     [output], layout=Layout(height="650px", border="2px solid gray")
    # )

    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)


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
    z = analytics(data, "Authors", top_by=None, top_n=None, limit_to=None, exclude=None)

    authors_dict = {
        author: num_docs
        for author, num_docs in zip(z.index, z.Num_Documents)
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
def __TAB2__(data):
    output = widgets.Output()
    with output:
        display(core_authors(data).head(50))
    grid = GridspecLayout(8, 8)
    grid[0:, 0:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )
    return grid


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
    m = analytics(
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


def __TAB3__(data):
    output = widgets.Output()
    with output:
        display(core_source_titles(data))
    grid = GridspecLayout(8, 8)
    grid[:, :] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )
    return grid


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
    data = data.sort_values(["Times_Cited", "Title"], ascending=[False, True])
    data = data[["Authors", "Year", "Title", "Source_title", "Times_Cited"]]
    data["Times_Cited"] = data.Times_Cited.map(lambda w: int(w))
    data = data.reset_index(drop=True)

    return data


def __TAB4__(data):
    """
    """
    output = widgets.Output()
    with output:
        display(top_documents(data))
    grid = GridspecLayout(10, 8)
    grid[0:, 0:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )
    return grid


###############################################################################
##
##  APP
##
###############################################################################


def app(data, limit_to=None, exclude=None, tab=None):
    return gui.APP(
        app_title="Analysis by Term",
        tab_titles=[
            "Term Analysis",
            "Worldmap",
            "Core Authors",
            "Core Source titles",
            "Top Documentes",
        ],
        tab_widgets=[
            __TAB0__(data, limit_to=limit_to, exclude=exclude),
            __TAB1__(data),
            __TAB2__(data),
            __TAB3__(data),
            __TAB4__(data),
        ],
        tab=tab,
    )

