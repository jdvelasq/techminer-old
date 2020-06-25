"""
Analysis by Term per Year
==================================================================================================


"""
import textwrap

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

import techminer.by_term as by_term
import techminer.by_year as by_year
import techminer.plots as plt
from techminer.explode import __explode
from techminer.keywords import Keywords
from techminer.params import EXCLUDE_COLS
from techminer.plots import COLORMAPS

TEXTLEN = 40

LEFT_PANEL_HEIGHT = "655px"
RIGHT_PANEL_WIDTH = "1200px"
PANE_HEIGHTS = ["80px", "720px", 0]


def analytics(
    data,
    column,
    output=0,
    top_by=None,
    top_n=None,
    sort_by=0,
    ascending=True,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    limit_to=None,
    exclude=None,
):
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
    ...          "Times_Cited": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year                     Authors  Times_Cited  ID
    0  2010  author 0;author 1;author 2           10   0
    1  2010                    author 0           11   1
    2  2011                    author 1           12   2
    3  2011                    author 3           13   3
    4  2012                    author 4           14   4
    5  2014                    author 4           15   5

    >>> analytics(df, 'Authors')[['Year', 'Authors', "Times_Cited", 'Num_Documents']]
       Year   Authors  Times_Cited  Num_Documents
    0  2010  author 0           21              2
    1  2010  author 1           10              1
    2  2010  author 2           10              1
    3  2011  author 1           12              1
    4  2011  author 3           13              1
    5  2012  author 4           14              1
    6  2014  author 4           15              1


    >>> analytics(df, 'Authors')[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Times_Cited']]
       Year   Authors  Perc_Num_Documents  Perc_Times_Cited
    0  2010  author 0               100.0            100.00
    1  2010  author 1                50.0             47.62
    2  2010  author 2                50.0             47.62
    3  2011  author 1                50.0             48.00
    4  2011  author 3                50.0             52.00
    5  2012  author 4               100.0            100.00
    6  2014  author 4               100.0            100.00

    >>> terms = ['author 1', 'author 2', 'author 3']
    >>> analytics(df, 'Authors', limit_to=terms)[['Year', 'Authors', "Times_Cited", 'Num_Documents', 'ID']]
       Year   Authors  Times_Cited  Num_Documents   ID
    0  2010  author 1           10              1  [0]
    1  2010  author 2           10              1  [0]
    2  2011  author 1           12              1  [2]
    3  2011  author 3           13              1  [3]

    >>> terms = ['author 1']
    >>> analytics(df, 'Authors', limit_to=terms)[['Year', 'Authors', "Times_Cited", 'Num_Documents', 'ID']]
       Year   Authors  Times_Cited  Num_Documents   ID
    0  2010  author 1           10              1  [0]
    1  2011  author 1           12              1  [2]


    >>> summary_by_term_per_year(df, 'Authors', exclude=terms)[['Year', 'Authors', 'Perc_Num_Documents', 'Perc_Times_Cited']]
       Year   Authors  Perc_Num_Documents  Perc_Times_Cited
    0  2010  author 0               100.0            100.00
    1  2010  author 2                50.0             47.62
    2  2011  author 3                50.0             52.00
    3  2012  author 4               100.0            100.00
    4  2014  author 4               100.0            100.00

    """

    #
    # Computation
    #
    x = data.copy()
    x = __explode(x[["Year", column, "Times_Cited", "ID"]], column)
    x["Num_Documents"] = 1
    result = x.groupby([column, "Year"], as_index=False).agg(
        {"Times_Cited": np.sum, "Num_Documents": np.size}
    )
    result = result.assign(
        ID=x.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
    )
    result["Times_Cited"] = result["Times_Cited"].map(lambda x: int(x))

    #
    # Indicators from scientoPy
    #
    summ = by_year.analytics(data)
    num_documents_by_year = {
        key: value for key, value in zip(summ.index, summ.Num_Documents)
    }
    times_cited_by_year = {
        key: value for key, value in zip(summ.index, summ.Times_Cited)
    }

    result["summary_documents_by_year"] = result.Year.apply(
        lambda w: num_documents_by_year[w]
    )
    result["summary_documents_by_year"] = result.summary_documents_by_year.map(
        lambda w: 1 if w == 0 else w
    )
    result["summary_times_cited_by_year"] = result.Year.apply(
        lambda w: times_cited_by_year[w]
    )
    result["summary_times_cited_by_year"] = result.summary_times_cited_by_year.map(
        lambda w: 1 if w == 0 else w
    )

    result["Perc_Num_Documents"] = 0.0
    result = result.assign(
        Perc_Num_Documents=round(
            result.Num_Documents / result.summary_documents_by_year * 100, 2
        )
    )

    result["Perc_Times_Cited"] = 0.0
    result = result.assign(
        Perc_Times_Cited=round(
            result.Times_Cited / result.summary_times_cited_by_year * 100, 2
        )
    )

    result.pop("summary_documents_by_year")
    result.pop("summary_times_cited_by_year")

    result = result.rename(
        columns={
            "Num_Documents": "Num_Documents_per_Year",
            "Times_Cited": "Times_Cited_per_Year",
            "Perc_Num_Documents": "%_Num_Documents_per_Year",
            "Perc_Times_Cited": "%_Times_Cited_per_Year",
        }
    )

    # ----------------------------------------------------------------------------------------

    #
    # top_by
    #
    if isinstance(top_by, str):
        top_by = top_by.replace(" ", "_")
        top_by = {
            "Num_Documents_per_Year": 0,
            "Times_Cited_per_Year": 1,
            "%_Num_Documents_per_Year": 2,
            "%_Times_Cited_per_Year": 3,
        }[top_by]

    # --------------------------------------------------------------------------------------

    #
    # Limit to
    #
    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        result = result[result[column].map(lambda w: w in limit_to)]

    #
    # Exclude
    #
    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        result = result[result[column].map(lambda w: w not in exclude)]

    # --------------------------------------------------------------------------------------
    if output == 0:

        columns = {
            0: ["Num_Documents_per_Year", "Times_Cited_per_Year"],
            1: ["Times_Cited_per_Year", "Num_Documents_per_Year"],
            2: ["%_Num_Documents_per_Year", "%_Times_Cited_per_Year"],
            3: ["%_Times_Cited_per_Year", "%_Num_Documents_per_Year"],
        }[top_by]

        result.sort_values(
            columns, ascending=False, inplace=True,
        )

        if top_n is not None:
            result = result.head(top_n)
            result = result.reset_index(drop=True)
        #
        # sort_by
        #
        if isinstance(sort_by, str):
            sort_by = sort_by.replace(" ", "_")
            sort_by = {
                "Alphabetic": 0,
                "Year": 1,
                "Num_Documents_per_Year": 2,
                "Times_Cited_per_Year": 3,
                "%_Num_Documents_per_Year": 4,
                "%_Times_Cited_per_Year": 5,
            }[sort_by]

        if isinstance(ascending, str):
            ascending = {"True": True, "False": False,}[ascending]

        if sort_by == 0:
            result = result.sort_values([column], ascending=ascending)
        else:
            result = result.sort_values(
                {
                    1: ["Year", "Num_Documents_per_Year", "Times_Cited_per_Year"],
                    2: ["Num_Documents_per_Year", "Times_Cited_per_Year", "Year"],
                    3: ["Times_Cited_per_Year", "Num_Documents_per_Year", "Year"],
                    4: ["%_Num_Documents_per_Year", "%_Times_Cited_per_Year", "Year"],
                    5: ["%_Times_Cited_per_Year", "%_Num_Documents_per_Year", "Year"],
                }[sort_by],
                ascending=ascending,
            )

        result.pop("ID")
        result = result[
            [
                column,
                "Year",
                "Num_Documents_per_Year",
                "Times_Cited_per_Year",
                "%_Num_Documents_per_Year",
                "%_Times_Cited_per_Year",
            ]
        ]
        return result

    # --------------------------------------------------------------------------------------
    if output in [1, 2, 3, 4]:

        selected_col = {
            0: "Num_Documents_per_Year",
            1: "Times_Cited_per_Year",
            2: "%_Num_Documents_per_Year",
            3: "%_Times_Cited_per_Year",
        }[top_by]

        for col in [
            "Num_Documents_per_Year",
            "Times_Cited_per_Year",
            "%_Num_Documents_per_Year",
            "%_Times_Cited_per_Year",
        ]:

            if col != selected_col:
                result.pop(col)

        result = pd.pivot_table(
            result, values=selected_col, index="Year", columns=column, fill_value=0,
        )

        max = result.max(axis=0)
        max = max.sort_values(ascending=False)
        if top_n is not None:
            max = max.head(top_n)
        result = result[max.index]

        sum_years = result.sum(axis=1)
        for year, index in zip(sum_years, sum_years.index):
            if year == 0:
                result = result.drop(axis=0, labels=index)
            else:
                break

        #
        # sort_by
        #
        if isinstance(sort_by, str):
            sort_by = sort_by.replace(" ", "_")
            sort_by = {"Alphabetic": 0, "Values": 1,}[sort_by]

        if sort_by == 0:
            columns = sorted(result.columns, reverse=not ascending)
        if sort_by == 1:
            columns = result.max(axis=0)
            columns = columns.sort_values(ascending=ascending)
            columns = columns.index.tolist()

        result = result[columns]

        #
        # Output
        #
        if output == 1:
            if cmap is None:
                return result
            else:
                return result.style.background_gradient(cmap=cmap, axis=None)

        if output == 2:
            return plt.heatmap(
                X=result.transpose(), cmap=cmap, figsize=figsize, fontsize=fontsize
            )

        if output == 3:
            return plt.bubble(
                X=result.transpose(),
                darkness=None,
                cmap=cmap,
                figsize=figsize,
                fontsize=fontsize,
            )

        if output == 4:
            return plt.gant(X=result, cmap=cmap, figsize=figsize, fontsize=fontsize,)

    # if sort_by == 0:
    #     result = result.sort_index(axis=1, ascending=ascending)

    # if sort_by == 1:
    #     selected_terms = summary_by_term.sort_values(
    #         ["Num_Documents", "Times_Cited"], ascending=ascending
    #     ).tolist()
    #     new_rows = [c for c in selected_terms if c in result.index.tolist()]
    #     result = result.loc[new_rows, :]

    # #  Convert to matrix
    # selected_col = {
    #     0: "Num_Documents",
    #     1: "Times_Cited",
    #     2: "Perc_Num_Documents",
    #     3: "Perc_Times_Cited",
    #     4: "Num_Documents",
    #     5: "Times_Cited",
    #     6: "Perc_Num_Documents",
    #     7: "Perc_Times_Cited",
    # }[top_by]

    ####################################

    #
    #
    #
    # summary_by_term = by_term.analytics(
    #     data=data,
    #     column=column,
    #     top_by=None,
    #     top_n=None,
    #     limit_to=limit_to,
    #     exclude=exclude,
    # )

    #
    # Limit to / Exclude
    #
    # filtered_terms = by_term.summary(
    #     data=x,
    #     column=column,
    #     top_by=None,
    #     top_n=None,
    #     limit_to=limit_to,
    #     exclude=exclude,
    # )[column].tolist()
    # result = result[result[column].map(lambda w: w in filtered_terms)]

    #
    # top_by and matrix conversion
    #

    # if isinstance(top_by, str):
    #     top_by = top_by.replace(" ", "_")
    #     top_by = {
    #         "Num_Documents": 0,
    #         "Times_Cited": 1,
    #         "%_Num_Documents": 2,
    #         "%_Times_Cited": 3,
    #         "Num_Documents_(Total)": 4,
    #         "Times_Cited_(Total)": 5,
    #         "%_Num_Documents_(Total)": 6,
    #         "%_Times_Cited_(Total)": 7,
    #     }[top_by]

    # columns = {
    #     0: ["Num_Documents", "Times_Cited"],
    #     1: ["Times_Cited", "Num_Documents"],
    #     2: ["Perc_Num_Documents", "Perc_Times_Cited"],
    #     3: ["Perc_Times_Cited", "Perc_Num_Documents"],
    #     4: ["Num_Documents", "Times_Cited"],
    #     5: ["Times_Cited", "Num_Documents"],
    #     6: ["Perc_Num_Documents", "Perc_Times_Cited"],
    #     7: ["Perc_Times_Cited", "Perc_Num_Documents"],
    # }[top_by]

    # #  Convert to matrix
    # selected_col = {
    #     0: "Num_Documents",
    #     1: "Times_Cited",
    #     2: "Perc_Num_Documents",
    #     3: "Perc_Times_Cited",
    #     4: "Num_Documents",
    #     5: "Times_Cited",
    #     6: "Perc_Num_Documents",
    #     7: "Perc_Times_Cited",
    # }[top_by]

    # for col in [
    #     "Num_Documents",
    #     "Times_Cited",
    #     "Perc_Num_Documents",
    #     "Perc_Times_Cited",
    # ]:
    #     if col != selected_col:
    #         result.pop(col)

    # result.sort_values(
    #     [selected_col, "Year", column], ascending=[False, True, True], inplace=True,
    # )

    # if top_n is not None:
    #     selected_terms = []
    #     n = top_n
    #     while len(selected_terms) < top_n:
    #         if top_by in [0, 1, 2, 3]:
    #             selected_terms = result.head(n)[column].tolist()
    #         else:
    #             if top_by in [4, 6]:
    #                 selected_terms = summary_by_term.sort_values(
    #                     ["Num_Documents", "Times_Cited"], ascending=False
    #                 )
    #                 selected_terms = selected_terms[column].head(n).tolist()
    #             else:
    #                 selected_terms = summary_by_term.sort_values(
    #                     ["Times_Cited", "Num_Documents"], ascending=False
    #                 )
    #                 selected_terms = selected_terms[column].head(n).tolist()
    #         selected_terms = list(set(selected_terms))
    #         n += 1

    #     result = result[result[column].map(lambda w: w in selected_terms)]

    # result.reset_index(drop=True)
    # result = pd.pivot_table(
    #     result, values=selected_col, index="Year", columns=column, fill_value=0,
    # )

    # result.columns = result.columns.tolist()
    # result.index = result.index.tolist()

    #
    # Sort by
    #
    # if isinstance(sort_by, str):
    #     sort_by = sort_by.replace(" ", "_")
    #     sort_by = {"Alphabetic": 0, "Num_Documents/Times_Cited": 1,}[sort_by]

    # if isinstance(ascending, str):
    #     ascending = {"True": True, "False": False,}[ascending]

    # if sort_by == 0:
    #     result = result.sort_index(axis=1, ascending=ascending)

    # if sort_by == 1:
    #     selected_terms = summary_by_term.sort_values(
    #         ["Num_Documents", "Times_Cited"], ascending=ascending
    #     ).tolist()
    #     new_rows = [c for c in selected_terms if c in result.index.tolist()]
    #     result = result.loc[new_rows, :]

    # #  Complete years
    # years = [year for year in range(result.index.min(), result.index.max() + 1)]
    # result = result.reindex(years, fill_value=0)

    # if output == 1:
    #     return plt.heatmap(result, cmap=cmap, figsize=figsize, fontsize=fontsize)

    # if output == 2:

    #     top_by_complement = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}[top_by]

    #     prop_to = analytics(
    #         data=data,
    #         column=column,
    #         output=0,
    #         top_by=top_by_complement,
    #         top_n=None,
    #         sort_by=sort_by,
    #         ascending=ascending,
    #         cmap=None,
    #         limit_to=result.columns.tolist(),
    #         exclude=None,
    #     )
    #     prop_to = prop_to.loc[
    #         result.index, [c for c in prop_to.columns if c in result.columns]
    #     ]

    #     return plt.bubble(
    #         result.transpose(),
    #         prop_to=prop_to.transpose(),
    #         cmap=cmap,
    #         figsize=figsize,
    #         fontsize=fontsize,
    #     )

    # if output == 3:
    #     return plt.plot(data=result, figsize=figsize, cmap=cmap, fontsize=12,)

    # if output == 4:
    #     return plt.gant_barh(
    #         data=result,
    #         height=0.6,
    #         left=None,
    #         align="center",
    #         cmap=cmap,
    #         figsize=figsize,
    #         fontsize=fontsize,
    #     )


##
##
##  Analytics by Value
##
##
def by_value_app(data, limit_to=None, exclude=None):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        #  0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Analytics",], layout=Layout(width="90%"),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="90%"),
            ),
        },
        # 2
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=[
                    "Num Documents per Year",
                    "Times Cited per Year",
                    "% Num Documents per Year",
                    "% Times Cited per Year",
                ],
                layout=Layout(width="90%"),
            ),
        },
        # 3
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(5, 101, 5)),
                ensure_option=True,
                layout=Layout(width="90%"),
            ),
        },
        # 4
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=[
                    "Alphabetic",
                    "Year",
                    "Num Documents per Year",
                    "Times Cited per Year",
                    "% Num Documents per Year",
                    "% Times Cited per Year",
                ],
                layout=Layout(width="90%"),
            ),
        },
        # 5
        {
            "arg": "ascending",
            "desc": "Ascending:",
            "widget": widgets.Dropdown(
                options=["True", "False",], layout=Layout(width="90%"),
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
        view = {"Analytics": 0,}[kwargs["view"]]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        sort_by = kwargs["sort_by"]
        ascending = {"True": True, "False": False}[kwargs["ascending"]]

        output.clear_output()
        with output:
            display(
                analytics(
                    data,
                    column=column,
                    output=view,
                    top_by=top_by,
                    top_n=top_n,
                    sort_by=sort_by,
                    ascending=ascending,
                    limit_to=limit_to,
                    exclude=exclude,
                    cmap=None,
                    figsize=None,
                )
            )
            return

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))

    grid = GridspecLayout(10, 8)

    grid[0, :] = widgets.HTML(
        value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
            "By Term per Year (Values)"
        )
    )
    #
    # Left panel
    #
    for index in range(len(left_panel)):
        grid[index + 1, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[index]["desc"]),
                left_panel[index]["widget"],
            ]
        )
    #
    # Output
    #
    grid[1:, 1:] = widgets.VBox(
        [output], layout=Layout(height="657px", border="2px solid gray")
    )

    return grid


##
##
##  Analytics by time matrix
##
##
def by_time_app(data, limit_to=None, exclude=None):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        #  0
        {
            "arg": "view",
            "desc": "View:",
            "widget": widgets.Dropdown(
                options=["Analytics", "Heatmap", "Bubble plot", "Gant"],
                layout=Layout(width="90%"),
            ),
        },
        # 1
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="90%"),
            ),
        },
        # 2
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=[
                    "Num Documents per Year",
                    "Times Cited per Year",
                    "% Num Documents per Year",
                    "% Times Cited per Year",
                ],
                layout=Layout(width="90%"),
            ),
        },
        # 3
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(5, 51, 5)),
                ensure_option=True,
                layout=Layout(width="90%"),
            ),
        },
        # 4
        {
            "arg": "sort_by",
            "desc": "Sort order:",
            "widget": widgets.Dropdown(
                options=["Alphabetic", "Values",], layout=Layout(width="90%"),
            ),
        },
        # 5
        {
            "arg": "ascending",
            "desc": "Ascending:",
            "widget": widgets.Dropdown(
                options=["True", "False",], layout=Layout(width="90%"),
            ),
        },
        # 6
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="90%"),),
        },
        # 7
        {
            "arg": "width",
            "desc": "Width:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), layout=Layout(width="90%"),
            ),
        },
        # 8
        {
            "arg": "height",
            "desc": "Height:",
            "widget": widgets.Dropdown(
                options=range(3, 15, 1), layout=Layout(width="90%"),
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
        view = {"Analytics": 1, "Heatmap": 2, "Bubble plot": 3, "Gant": 4}[
            kwargs["view"]
        ]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        sort_by = kwargs["sort_by"]
        ascending = {"True": True, "False": False}[kwargs["ascending"]]
        cmap = kwargs["cmap"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        output.clear_output()
        with output:
            display(
                analytics(
                    data,
                    column=column,
                    output=view,
                    top_by=top_by,
                    top_n=top_n,
                    sort_by=sort_by,
                    ascending=ascending,
                    limit_to=limit_to,
                    exclude=exclude,
                    cmap=cmap,
                    figsize=(width, height),
                )
            )
            return

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    #
    grid = GridspecLayout(10, 8)
    #
    grid[0, :] = widgets.HTML(
        value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
            "By Term per Year (Time)"
        )
    )
    #
    # Left panel
    #
    for index in range(len(left_panel)):
        grid[index + 1, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[index]["desc"]),
                left_panel[index]["widget"],
            ]
        )
    #
    # Output
    #
    grid[1:, 1:] = widgets.VBox(
        [output], layout=Layout(height="657px", border="2px solid gray")
    )

    return grid


#
# def gant(x, column, limit_to=None, exclude=None):
#     """Computes the number of documents by term per year.
#
#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.
#         as_matrix (bool): Results are returned as a matrix.
#         keywords (Keywords): filter the result using the specified Keywords object.
#
#     Returns:
#         DataFrame.
#
#
#     Examples
#     ----------------------------------------------------------------------------------------------
#
#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2011, 2011, 2012, 2015, 2012, 2016],
#     ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 3,author 4,author 4".split(","),
#     ...          "Times_Cited": list(range(10,17)),
#     ...          "ID": list(range(7)),
#     ...     }
#     ... )
#     >>> num_documents_by_term_per_year(df, 'Authors', as_matrix=True)
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         1         1         0         0
#     2011         1         1         0         0         0
#     2012         0         0         0         1         1
#     2015         0         0         0         1         0
#     2016         0         0         0         0         1
#
#     >>> gant(df, 'Authors')
#           author 0  author 1  author 2  author 3  author 4
#     2010         1         1         1         0         0
#     2011         1         1         0         0         0
#     2012         0         0         0         1         1
#     2013         0         0         0         1         1
#     2014         0         0         0         1         1
#     2015         0         0         0         1         1
#     2016         0         0         0         0         1
#
#     >>> terms = Keywords(['author 1', 'author 2'])
#     >>> gant(df, 'Authors', limit_to=terms)
#           author 1  author 2
#     2010         1         1
#     2011         1         0
#
#     >>> gant(df, 'Authors', exclude=terms)
#           author 0  author 3  author 4
#     2010         1         0         0
#     2011         1         0         0
#     2012         0         1         1
#     2013         0         1         1
#     2014         0         1         1
#     2015         0         1         1
#     2016         0         0         1
#
#     """
#
#     years = [year for year in range(result.index.min(), result.index.max() + 1)]
#     result = result.reindex(years, fill_value=0)
#     matrix1 = result.copy()
#     matrix1 = matrix1.cumsum()
#     matrix1 = matrix1.applymap(lambda x: True if x > 0 else False)
#     matrix2 = result.copy()
#     matrix2 = matrix2.sort_index(ascending=False)
#     matrix2 = matrix2.cumsum()
#     matrix2 = matrix2.applymap(lambda x: True if x > 0 else False)
#     matrix2 = matrix2.sort_index(ascending=True)
#     result = matrix1.eq(matrix2)
#     result = result.applymap(lambda x: 1 if x is True else 0)
#     return result
#

##
##
## APP
##
##


##
##
##  Growth Indicators
##
##


def growth_indicators(x, column, timewindow=2, top_n=None, limit_to=None, exclude=None):
    """Computes the average growth rate of a group of terms.

    Args:
        column (str): the column to explode.
        timewindow (int): time window for analysis
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.DataFrame(
    ...   {
    ...     "Year": [2010, 2010, 2011, 2011, 2012, 2013, 2014, 2014],
    ...     "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4,author 0;author 3,author 3;author 4".split(","),
    ...     "Times_Cited": list(range(10,18)),
    ...     "ID": list(range(8)),
    ...   }
    ... )
    >>> x
       Year                     Authors  Times_Cited  ID
    0  2010  author 0;author 1;author 2           10   0
    1  2010                    author 0           11   1
    2  2011                    author 1           12   2
    3  2011                    author 3           13   3
    4  2012                    author 4           14   4
    5  2013                    author 4           15   5
    6  2014           author 0;author 3           16   6
    7  2014           author 3;author 4           17   7

    >>> num_documents_by_term_per_year(x, 'Authors', as_matrix=True)
          author 0  author 1  author 2  author 3  author 4
    2010         2         1         1         0         0
    2011         0         1         0         1         0
    2012         0         0         0         0         1
    2013         0         0         0         0         1
    2014         1         0         0         2         1

    >>> growth_indicators(x, 'Authors')
        Authors       AGR  ADY   PDLY  Before 2013  Between 2013-2014
    0  author 3  0.666667  1.0  12.50            1                  2
    1  author 0  0.333333  0.5   6.25            2                  1
    2  author 4  0.000000  1.0  12.50            1                  2

    >>> terms = ['author 3', 'author 4']
    >>> growth_indicators(x, 'Authors', limit_to=terms)
        Authors       AGR  ADY  PDLY  Before 2013  Between 2013-2014
    0  author 3  0.666667  1.0  12.5            1                  2
    1  author 4  0.000000  1.0  12.5            1                  2

    >>> growth_indicators(x, 'Authors', exclude=terms)
        Authors       AGR  ADY  PDLY  Before 2011  Between 2011-2014
    0  author 1 -0.333333  0.5  6.25            1                  1
    1  author 0 -0.333333  0.5  6.25            2                  1

    """

    def compute_agr():
        result = num_documents_by_term_per_year(
            x, column=column, limit_to=limit_to, exclude=exclude
        )
        years_agr = sorted(set(result.Year))[-(timewindow + 1) :]
        years_agr = [years_agr[0], years_agr[-1]]
        result = result[result.Year.map(lambda w: w in years_agr)]
        result.pop("ID")
        result = pd.pivot_table(
            result, columns="Year", index=column, values="Num_Documents", fill_value=0,
        )
        result["AGR"] = 0.0
        result = result.assign(
            AGR=(result[years_agr[1]] - result[years_agr[0]]) / (timewindow + 1)
        )
        result.pop(years_agr[0])
        result.pop(years_agr[1])
        result.columns = list(result.columns)
        result = result.sort_values(by=["AGR", column], ascending=False)
        result.reset_index(drop=True)
        return result

    def compute_ady():
        result = num_documents_by_term_per_year(
            x, column=column, limit_to=limit_to, exclude=exclude
        )
        years_ady = sorted(set(result.Year))[-timewindow:]
        result = result[result.Year.map(lambda w: w in years_ady)]
        result = result.groupby([column], as_index=False).agg({"Num_Documents": np.sum})
        result = result.rename(columns={"Num_Documents": "ADY"})
        result["ADY"] = result.ADY.map(lambda w: w / timewindow)
        result = result.reset_index(drop=True)
        return result

    def compute_num_documents():
        result = num_documents_by_term_per_year(
            x, column=column, limit_to=limit_to, exclude=exclude
        )
        years_between = sorted(set(result.Year))[-timewindow:]
        years_before = sorted(set(result.Year))[0:-timewindow]
        between = result[result.Year.map(lambda w: w in years_between)]
        before = result[result.Year.map(lambda w: w in years_before)]
        between = between.groupby([column], as_index=False).agg(
            {"Num_Documents": np.sum}
        )
        between = between.rename(
            columns={
                "Num_Documents": "Between {}-{}".format(
                    years_between[0], years_between[-1]
                )
            }
        )
        before = before.groupby([column], as_index=False).agg({"Num_Documents": np.sum})
        before = before.rename(
            columns={"Num_Documents": "Before {}".format(years_between[0])}
        )
        result = pd.merge(before, between, on=column)
        return result

    result = compute_agr()
    ady = compute_ady()
    result = pd.merge(result, ady, on=column)
    result = result.assign(PDLY=round(result.ADY / len(x) * 100, 2))
    num_docs = compute_num_documents()
    result = pd.merge(result, num_docs, on=column)
    result = result.reset_index(drop=True)
    return result


def growth_app(x, limit_to=None, exclude=None):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in x.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        # 0
        {
            "arg": "term",
            "desc": "Term to analyze:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in x.columns],
                ensure_option=True,
                layout=Layout(width="90%"),
            ),
        },
        # 1
        {
            "arg": "analysis_type",
            "desc": "Analysis type:",
            "widget": widgets.Dropdown(
                options=[
                    "Average Growth Rate",
                    "Average Documents per Year",
                    "Percentage of Documents in Last Years",
                    "Number of Document Published",
                ],
                value="Average Growth Rate",
                layout=Layout(width="90%"),
            ),
        },
        # 2
        {
            "arg": "time_window",
            "desc": "Time window:",
            "widget": widgets.Dropdown(
                options=["2", "3", "4", "5"], value="2", layout=Layout(width="90%"),
            ),
        },
        # 3
        {
            "arg": "plot_type",
            "desc": "Plot type:",
            "widget": widgets.Dropdown(
                options=["bar", "barh"], layout=Layout(width="90%"),
            ),
        },
        # 4
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width="90%"),
            ),
        },
        # 5
        {
            "arg": "top_n",
            "desc": "Top N:",
            "widget": widgets.Dropdown(
                options=list(range(5, 51, 5)),
                ensure_option=True,
                layout=Layout(width="90%"),
            ),
        },
        # 6
        {
            "arg": "figsize_width",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), layout=Layout(width="90%"),
            ),
        },
        # 7
        {
            "arg": "figsize_height",
            "desc": "Figsize",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), layout=Layout(width="90%"),
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
        cmap = kwargs["cmap"]
        analysis_type = kwargs["analysis_type"]
        top_n = kwargs["top_n"]
        plot_type = kwargs["plot_type"]
        time_window = int(kwargs["time_window"])
        figsize_width = int(kwargs["figsize_width"])
        figsize_height = int(kwargs["figsize_height"])
        #
        plots = {"bar": plt.bar, "barh": plt.barh}
        plot = plots[plot_type]
        #
        df = growth_indicators(
            x, term, timewindow=time_window, limit_to=limit_to, exclude=exclude
        )
        output.clear_output()

        with output:
            if analysis_type == "Average Growth Rate":
                df = df.sort_values("AGR", ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                display(
                    plot(
                        df[[term, "AGR"]],
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if analysis_type == "Average Documents per Year":
                df = df.sort_values("ADY", ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                display(
                    plot(
                        df[[term, "ADY"]],
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if analysis_type == "Percentage of Documents in Last Years":
                df = df.sort_values("PDLY", ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                display(
                    plot(
                        df[[term, "PDLY"]],
                        cmap=cmap,
                        figsize=(figsize_width, figsize_height),
                    )
                )
            if analysis_type == "Number of Document Published":
                df["Num_Documents"] = df[df.columns[-2]] + df[df.columns[-1]]
                df = df.sort_values("Num_Documents", ascending=False).head(top_n)
                df = df.reset_index(drop=True)
                df.pop("Num_Documents")
                if plot_type == "bar":
                    display(
                        plt.stacked_bar(
                            df[[term, df.columns[-2], df.columns[-1]]],
                            figsize=(figsize_width, figsize_height),
                            cmap=cmap,
                        )
                    )
                if plot_type == "barh":
                    display(
                        plt.stacked_barh(
                            df[[term, df.columns[-2], df.columns[-1]]],
                            figsize=(figsize_width, figsize_height),
                            cmap=cmap,
                        )
                    )

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))

    grid = GridspecLayout(10, 8)

    grid[0, :] = widgets.HTML(
        value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
            "Growth Indicators"
        )
    )

    #
    # Left panel
    #
    for index in range(len(left_panel)):
        grid[index + 1, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[index]["desc"]),
                left_panel[index]["widget"],
            ]
        )
    #
    # Output
    #
    grid[1:, 1:] = widgets.VBox([output], layout=Layout(border="2px solid gray"))

    return grid


def app(df, limit_to=None, exclude=None):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [
        __APP0__(df, limit_to, exclude),
        __APP1__(df, limit_to, exclude),
        __APP2__(df, limit_to, exclude),
    ]
    body.set_title(0, "Analysis by Value")
    body.set_title(1, "Analysis by Time")
    body.set_title(2, "Growth Indicators")
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
