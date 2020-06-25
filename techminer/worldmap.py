import techminer.plots as plt
import ipywidgets as widgets

from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout
from techminer.plots import COLORMAPS
from techminer import by_term


def app(data, limit_to=None, exclude=None):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    left_panel = [
        # 0
        {
            "arg": "column",
            "desc": "Column to analyze:",
            "widget": widgets.Dropdown(
                options=["Countries", "Country_1st_Author"], layout=Layout(width="90%"),
            ),
        },
        # 1
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num_Documents", "Times_Cited"], layout=Layout(width="90%"),
            ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, disable=False, layout=Layout(width="90%"),
            ),
        },
        # 3
        {
            "arg": "width",
            "desc": "Width",
            "widget": widgets.Dropdown(
                options=range(15, 21, 1),
                ensure_option=True,
                layout=Layout(width="90%"),
            ),
        },
        # 4
        {
            "arg": "height",
            "desc": "Height",
            "widget": widgets.Dropdown(
                options=range(4, 9, 1), ensure_option=True, layout=Layout(width="90%"),
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
        result = by_term.analytics(
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
    args = {control["arg"]: control["widget"] for control in left_panel}
    output = widgets.Output()
    with output:
        display(widgets.interactive_output(server, args,))
    #
    grid = GridspecLayout(10, 8)
    #
    grid[0, :] = widgets.HTML(
        value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
            "Woldmap"
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

