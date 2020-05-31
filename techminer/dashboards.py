"""
Jupyter Lab Interface
==================================================================================================


"""

import ipywidgets as widgets
from ipywidgets import Layout, AppLayout

from IPython.display import display, HTML, clear_output

import techminer.analytics as tc
import techminer.plots as plt

FIGSIZE = (12, 6)
PANEL_HEIGHT = "400px"

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
    "Institution 1st" "Institutions",
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

    def compute(selected, plot_type, cmap):
        df = tc.summary_by_year(x)
        df = df[data[selected]]
        plot = plots[plot_type]
        output.clear_output()
        with output:
            display(plot(df, cmap=cmap, figsize=FIGSIZE))

    #
    # Options
    #
    data = {
        "Documents by Year": ["Year", "Num Documents"],
        "Cum. Documents by Year": ["Year", "Num Documents (Cum)"],
        "Times Cited by Year": ["Year", "Cited by"],
        "Cum. Times Cited by Year": ["Year", "Cited by (Cum)"],
        "Avg. Times Cited by Year": ["Year", "Avg. Cited by"],
    }
    plots = {"bar": plt.bar, "barh": plt.barh}
    #
    selected = widgets.Dropdown(
        options=list(data.keys()), value=list(data.keys())[0], disable=False,
    )
    plot_type = widgets.Dropdown(options=["bar", "barh"], disable=False,)
    cmap = widgets.Dropdown(options=COLORMAPS, disable=False,)
    #
    output = widgets.Output()
    with output:
        display(
            widgets.interactive_output(
                compute, {"selected": selected, "plot_type": plot_type, "cmap": cmap}
            )
        )
    #
    left_box = widgets.VBox(
        [
            widgets.VBox([widgets.Label(value="Plot"), selected]),
            widgets.VBox([widgets.Label(value="Plot type:"), plot_type]),
            widgets.VBox([widgets.Label(value="Colormap:"), cmap]),
        ],
        layout=Layout(height=PANEL_HEIGHT, border="1px solid gray"),
    )
    right_box = widgets.VBox([output])

    return AppLayout(
        header=widgets.HTML(value=html_title("Summary by Year")),
        left_sidebar=left_box,
        center=right_box,
        right_sidebar=None,
        pane_widths=[2, 5, 0],
        pane_heights=["85px", 5, 0],
    )


def summary_by_term(x):
    """ Summary by Term dashboard.
    
    Args:
        df (pandas.DataFrame): bibliographic dataframe.
    
    """

    def tab_term_plots():
        def compute_by_term(term, top_n, analysis_type, plot_type, cmap):
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

        columns = [z for z in COLUMNS if z in x.columns]
        term = widgets.Select(options=columns, ensure_option=True, disabled=False,)

        analysis_type = widgets.Dropdown(
            options=["Frequency", "Citation"], value="Frequency", disable=False,
        )
        #
        plots = {"bar": plt.bar, "barh": plt.barh, "pie": plt.pie}
        plot_type = widgets.Dropdown(options=list(plots.keys()), disable=False,)
        cmap = widgets.Dropdown(options=COLORMAPS, disable=False,)
        #
        top_n = widgets.IntSlider(
            value=10,
            min=10,
            max=50,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        #
        output = widgets.Output()
        with output:
            display(
                widgets.interactive_output(
                    compute_by_term,
                    {
                        "term": term,
                        "top_n": top_n,
                        "analysis_type": analysis_type,
                        "plot_type": plot_type,
                        "cmap": cmap,
                    },
                )
            )
        #
        left_box = widgets.VBox(
            [
                widgets.VBox([widgets.Label(value="Term to analyze:"), term]),
                widgets.VBox([widgets.Label(value="Analysis type:"), analysis_type]),
                widgets.VBox([widgets.Label(value="Top n terms:"), top_n]),
                widgets.VBox([widgets.Label(value="Plot type:"), plot_type]),
                widgets.VBox([widgets.Label(value="Colormap:"), cmap]),
            ],
            layout=Layout(height="400px", border="1px solid gray"),
        )
        right_box = widgets.VBox([output])
        return widgets.HBox([left_box, right_box])

    def tab_worldmap():
        def compute_worldmap(term, analysis_type, cmap):
            df = tc.summary_by_term(x, term)
            if analysis_type == "Frequency":
                df = df[[term, "Num Documents"]]
            else:
                df = df[[term, "Cited by"]]
            df = df.reset_index(drop=True)
            output.clear_output()
            with output:
                display(plt.worldmap(df, figsize=FIGSIZE, cmap=cmap))

        term = widgets.Select(
            options=["Countries", "Country 1st"], ensure_option=True, disabled=False,
        )
        analysis_type = widgets.Dropdown(
            options=["Frequency", "Citation"], value="Frequency", disable=False,
        )
        cmap = widgets.Dropdown(options=COLORMAPS, disable=False,)
        #
        output = widgets.Output()
        with output:
            display(
                widgets.interactive_output(
                    compute_worldmap,
                    {"term": term, "analysis_type": analysis_type, "cmap": cmap,},
                )
            )
        #
        left_box = widgets.VBox(
            [
                widgets.VBox([widgets.Label(value="Term to analyze:"), term]),
                widgets.VBox([widgets.Label(value="Analysis type:"), analysis_type]),
                widgets.VBox([widgets.Label(value="Colormap:"), cmap]),
            ],
            layout=Layout(height=PANEL_HEIGHT, border="1px solid gray"),
        )
        right_box = widgets.VBox([output])
        return widgets.HBox([left_box, right_box])

    #
    tab_nest = widgets.Tab()
    tab_nest.children = [tab_term_plots(), tab_worldmap()]
    tab_nest.set_title(0, "Term analysis")
    tab_nest.set_title(1, "Worldmap")

    return AppLayout(
        header=widgets.HTML(value=html_title("Summary by Term")),
        left_sidebar=None,
        center=tab_nest,
        right_sidebar=None,
        pane_widths=[0, 5, 0],
        pane_heights=["85px", 5, 0],
    )


def summary_by_term_per_year(x):
    def compute_by_term(term, top_n, analysis_type, plot_type, cmap):
        plot = plt.heatmap if plot_type == "Heatmap" else plt.gant
        if analysis_type == "Frequency":
            top = tc.documents_by_term(x, term).head(top_n)[term].tolist()
            matrix = tc.documents_by_term_per_year(x, term, as_matrix=True)
        else:
            top = tc.citations_by_term(x, term).head(top_n)[term].tolist()
            matrix = tc.citations_by_term_per_year(x, term, as_matrix=True)
        print(matrix.columns)
        matrix = matrix[top]
        output.clear_output()
        with output:
            if plot_type == "Heatmap":
                display(plot(matrix, cmap=cmap, figsize=FIGSIZE))
            else:
                display(plot(matrix, figsize=FIGSIZE))

    #
    columns = [z for z in COLUMNS if z in x.columns]
    term = widgets.Select(options=columns, ensure_option=True, disabled=False,)
    analysis_type = widgets.Dropdown(
        options=["Frequency", "Citation"], value="Frequency", disable=False,
    )
    plots = {"Heatmap": plt.heatmap, "Gant": plt.gant}
    plot_type = widgets.Dropdown(options=list(plots.keys()), disable=False,)
    cmap = widgets.Dropdown(options=COLORMAPS, disable=False,)
    #
    top_n = widgets.IntSlider(
        value=10,
        min=10,
        max=30,
        step=1,
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )
    #
    output = widgets.Output()
    with output:
        display(
            widgets.interactive_output(
                compute_by_term,
                {
                    "term": term,
                    "top_n": top_n,
                    "analysis_type": analysis_type,
                    "plot_type": plot_type,
                    "cmap": cmap,
                },
            )
        )
    #
    left_box = widgets.VBox(
        [
            widgets.VBox([widgets.Label(value="Term to analyze:"), term]),
            widgets.VBox([widgets.Label(value="Analysis type:"), analysis_type]),
            widgets.VBox([widgets.Label(value="Top n terms:"), top_n]),
            widgets.VBox([widgets.Label(value="Plot type:"), plot_type]),
            widgets.VBox([widgets.Label(value="Colormap:"), cmap]),
        ],
        layout=Layout(height=PANEL_HEIGHT, border="1px solid gray"),
    )
    right_box = widgets.VBox([output])

    return AppLayout(
        header=widgets.HTML(value=html_title("Summary by Term per Year")),
        left_sidebar=left_box,
        center=right_box,
        right_sidebar=None,
        pane_widths=[2, 5, 0],
        pane_heights=["85px", 5, 0],
    )
