"""
Jupyter Lab Interface
==================================================================================================


"""

import ipywidgets as widgets
from ipywidgets import Layout, AppLayout

from IPython.display import display, HTML, clear_output

import techminer.analytics as tc
import techminer.plots as plt

FIGSIZE = (18, 9.1)
LEFT_PANEL_HEIGHT = "588px"
PANE_HEIGHTS = ["80px", "650px", 0]
WIDGET_WIDTH = "200px"


COLORMAPS = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]

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


def html_title(x):
    return (
        "<h1>{}</h1>".format(x)
        + "<hr style='height:2px;border-width:0;color:gray;background-color:gray'>"
    )


def summary_by_year(x):
    """ Summary by year dashboard.
    
    Args:
        df (pandas.DataFrame): bibliographic dataframe.
    
    """

    def tab_0():
        #
        def compute(selected_plot, plot_type, cmap):
            #
            plots = {"bar": plt.bar, "barh": plt.barh}
            data = {
                "Documents by Year": ["Year", "Num Documents"],
                "Cum. Documents by Year": ["Year", "Num Documents (Cum)"],
                "Times Cited by Year": ["Year", "Cited by"],
                "Cum. Times Cited by Year": ["Year", "Cited by (Cum)"],
                "Avg. Times Cited by Year": ["Year", "Avg. Cited by"],
            }
            #
            df = tc.summary_by_year(x)
            df = df[data[selected_plot]]
            plot = plots[plot_type]
            output.clear_output()
            with output:
                display(plot(df, cmap=cmap, figsize=FIGSIZE))

        #
        LEFT_PANEL = [
            (
                "Plot:",
                "selected_plot",
                widgets.Dropdown(
                    options=[
                        "Documents by Year",
                        "Cum. Documents by Year",
                        "Times Cited by Year",
                        "Cum. Times Cited by Year",
                        "Avg. Times Cited by Year",
                    ],
                    value="Documents by Year",
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Plot type:",
                "plot_type",
                widgets.Dropdown(
                    options=["bar", "barh"],
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Colormap:",
                "cmap",
                widgets.Dropdown(
                    options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
        ]
        #
        args = {key: value for _, key, value in LEFT_PANEL}
        output = widgets.Output()
        with output:
            display(widgets.interactive_output(compute, args,))
        return widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.VBox([widgets.Label(value=text), widget])
                        for text, _, widget in LEFT_PANEL
                    ],
                    layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
                ),
                widgets.VBox([output]),
            ]
        )

    #
    #
    body = widgets.Tab()
    body.children = [tab_0()]
    #
    body.set_title(0, "Time analysis")

    return AppLayout(
        header=widgets.HTML(value=html_title("Summary by Year")),
        left_sidebar=None,
        center=body,
        right_sidebar=None,
        pane_heights=PANE_HEIGHTS,
    )


def summary_by_term(x):
    """ Summary by Term dashboard.
    
    Args:
        df (pandas.DataFrame): bibliographic dataframe.
    
    """

    def tab_0():
        #
        def compute(term, analysis_type, plot_type, cmap, top_n):
            #
            plots = {"bar": plt.bar, "barh": plt.barh, "pie": plt.pie}
            #
            df = tc.summary_by_term(x, term)
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
                display(plot(df, figsize=FIGSIZE, cmap=cmap))

        #
        LEFT_PANEL = [
            (
                "Term to analyze:",
                "term",
                widgets.Select(
                    options=[z for z in COLUMNS if z in x.columns],
                    ensure_option=True,
                    disabled=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Analysis type:",
                "analysis_type",
                widgets.Dropdown(
                    options=["Frequency", "Citation"],
                    value="Frequency",
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Plot type:",
                "plot_type",
                widgets.Dropdown(
                    options=["bar", "barh", "pie"],
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Colormap:",
                "cmap",
                widgets.Dropdown(
                    options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Top N:",
                "top_n",
                widgets.IntSlider(
                    value=10,
                    min=10,
                    max=50,
                    step=1,
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
        ]
        #
        args = {key: value for _, key, value in LEFT_PANEL}
        output = widgets.Output()
        with output:
            display(widgets.interactive_output(compute, args,))
        return widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.VBox([widgets.Label(value=text), widget])
                        for text, _, widget in LEFT_PANEL
                    ],
                    layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
                ),
                widgets.VBox([output]),
            ]
        )

    def tab_1():
        #
        def compute(term, analysis_type, cmap):
            df = tc.summary_by_term(x, term)
            if analysis_type == "Frequency":
                df = df[[term, "Num Documents"]]
            else:
                df = df[[term, "Cited by"]]
            df = df.reset_index(drop=True)
            output.clear_output()
            with output:
                display(plt.worldmap(df, figsize=FIGSIZE, cmap=cmap))

        #
        LEFT_PANEL = [
            (
                "Term to analyze:",
                "term",
                widgets.Select(
                    options=["Countries", "Country 1st"],
                    ensure_option=True,
                    disabled=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Analysis type:",
                "analysis_type",
                widgets.Dropdown(
                    options=["Frequency", "Citation"],
                    value="Frequency",
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Colormap:",
                "cmap",
                widgets.Dropdown(
                    options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
        ]
        #
        args = {key: value for _, key, value in LEFT_PANEL}
        output = widgets.Output()
        with output:
            display(widgets.interactive_output(compute, args,))
        return widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.VBox([widgets.Label(value=text), widget])
                        for text, _, widget in LEFT_PANEL
                    ],
                    layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
                ),
                widgets.VBox([output]),
            ]
        )

    #
    #
    body = widgets.Tab()
    body.children = [tab_0(), tab_1()]
    #
    body.set_title(0, "Time analysis")
    body.set_title(1, "Worldmap")

    return AppLayout(
        header=widgets.HTML(value=html_title("Summary by Term")),
        left_sidebar=None,
        center=body,
        right_sidebar=None,
        pane_heights=PANE_HEIGHTS,
    )


def summary_by_term_per_year(x):
    #
    def tab_0():
        #
        def compute(term, analysis_type, plot_type, cmap, top_n):
            #
            plots = {"Heatmap": plt.heatmap, "Gant": plt.gant}
            plot = plots[plot_type]
            #
            if analysis_type == "Frequency":
                top = tc.documents_by_term(x, term).head(top_n)[term].tolist()
                matrix = tc.documents_by_term_per_year(x, term, as_matrix=True)
            else:
                top = tc.citations_by_term(x, term).head(top_n)[term].tolist()
                matrix = tc.citations_by_term_per_year(x, term, as_matrix=True)
            matrix = matrix[top]
            output.clear_output()
            with output:
                if plot_type == "Heatmap":
                    display(plot(matrix, cmap=cmap, figsize=FIGSIZE))
                if plot_type == "Gant":
                    display(plot(matrix, figsize=FIGSIZE))

        #
        LEFT_PANEL = [
            (
                "Term to analyze:",
                "term",
                widgets.Select(
                    options=[z for z in COLUMNS if z in x.columns],
                    ensure_option=True,
                    disabled=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Analysis type:",
                "analysis_type",
                widgets.Dropdown(
                    options=["Frequency", "Citation"],
                    value="Frequency",
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Plot type:",
                "plot_type",
                widgets.Dropdown(
                    options=["Heatmap", "Gant"],
                    disable=False,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Colormap:",
                "cmap",
                widgets.Dropdown(
                    options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Top N:",
                "top_n",
                widgets.IntSlider(
                    value=10,
                    min=10,
                    max=50,
                    step=1,
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
        ]
        #
        args = {key: value for _, key, value in LEFT_PANEL}
        output = widgets.Output()
        with output:
            display(widgets.interactive_output(compute, args,))
        return widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.VBox([widgets.Label(value=text), widget])
                        for text, _, widget in LEFT_PANEL
                    ],
                    layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
                ),
                widgets.VBox([output]),
            ]
        )

    #
    #
    body = widgets.Tab()
    body.children = [tab_0()]
    #
    body.set_title(0, "Heatmap")

    return AppLayout(
        header=widgets.HTML(value=html_title("Summary by Term per Year")),
        left_sidebar=None,
        center=body,
        right_sidebar=None,
        pane_heights=PANE_HEIGHTS,
    )


def co_occurrence_analysis(x):
    #
    def tab_0():
        #
        def compute(column, by, min_value, cmap):
            #
            matrix = tc.co_occurrence(
                x,
                column=column,
                by=by,
                as_matrix=True,
                min_value=min_value,
                keywords=None,
            )
            if LEFT_PANEL[3][2].value > matrix.max().max():
                LEFT_PANEL[3][2].value = matrix.max().max()
            LEFT_PANEL[3][2].max = matrix.max().max()
            #
            output.clear_output()
            with output:
                if len(matrix.columns) < 51 and len(matrix.index) < 51:
                    display(matrix.style.background_gradient(cmap=cmap))
                else:
                    display(matrix)

        #
        LEFT_PANEL = [
            (
                "Term to analyze:",
                "column",
                widgets.Select(
                    options=[z for z in COLUMNS if z in x.columns],
                    ensure_option=True,
                    disabled=False,
                    continuous_update=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "By:",
                "by",
                widgets.Select(
                    options=[z for z in COLUMNS if z in x.columns],
                    ensure_option=True,
                    disabled=False,
                    continuous_update=True,
                    layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Colormap:",
                "cmap",
                widgets.Dropdown(
                    options=COLORMAPS, disable=False, layout=Layout(width=WIDGET_WIDTH),
                ),
            ),
            (
                "Min value:",
                "min_value",
                widgets.IntSlider(
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
            ),
        ]
        #
        args = {key: value for _, key, value in LEFT_PANEL}
        output = widgets.Output()
        with output:
            display(widgets.interactive_output(compute, args,))
        return widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.VBox([widgets.Label(value=text), widget])
                        for text, _, widget in LEFT_PANEL
                    ],
                    layout=Layout(
                        height=LEFT_PANEL_HEIGHT, width="210px", border="1px solid gray"
                    ),
                ),
                widgets.VBox([output], layout=Layout(width="1000px")),
            ]
        )

    #
    #
    body = widgets.Tab()
    body.children = [tab_0()]
    #
    body.set_title(0, "Heatmap")
    return AppLayout(
        header=widgets.HTML(value=html_title("Co-occurrence Analysis")),
        left_sidebar=None,
        center=body,
        right_sidebar=None,
        pane_heights=PANE_HEIGHTS,
    )
