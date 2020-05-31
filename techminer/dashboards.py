"""
Jupyter Lab Interface
==================================================================================================


"""

import ipywidgets as widgets
from ipywidgets import Layout, AppLayout

# from IPython.display import display, HTML, clear_output

import techminer.analytics as tc
import techminer.plots as plt

FIGSIZE = (12, 6)
PANEL_HEIGHT = "400px"


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
