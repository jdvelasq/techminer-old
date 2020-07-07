from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

from techminer.plots import COLORMAPS
import ipywidgets as widgets

#
# Common controls GUI definition
#
def ascending():
    return {
        "arg": "ascending",
        "desc": "Ascending:",
        "widget": widgets.Dropdown(options=[True, False], layout=Layout(width="55%"),),
    }


def cmap(arg="cmap", desc="Colormap:"):
    return {
        "arg": arg,
        "desc": desc,
        "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
    }


def dropdown(desc, options):
    arg = desc.replace(":", "").replace(" ", "_").lower()
    return {
        "arg": arg,
        "desc": desc,
        "widget": widgets.Dropdown(options=options, layout=Layout(width="55%"),),
    }


def fig_height():
    return {
        "arg": "height",
        "desc": "Height:",
        "widget": widgets.Dropdown(
            options=range(5, 15, 1), layout=Layout(width="55%"),
        ),
    }


def fig_width():
    return {
        "arg": "width",
        "desc": "Width:",
        "widget": widgets.Dropdown(
            options=range(5, 15, 1), layout=Layout(width="55%"),
        ),
    }


def linkage():
    return {
        "arg": "linkage",
        "desc": "Linkage:",
        "widget": widgets.Dropdown(
            options=["ward", "complete", "average", "single"],
            layout=Layout(width="55%"),
        ),
    }


def n_clusters():
    return {
        "arg": "n_clusters",
        "desc": "# Clusters:",
        "widget": widgets.Dropdown(
            options=list(range(2, 21)), layout=Layout(width="55%"),
        ),
    }


def n_components():
    return {
        "arg": "n_components",
        "desc": "# Comp. :",
        "widget": widgets.Dropdown(
            options=list(range(2, 21)), layout=Layout(width="55%"),
        ),
    }


def normalization():
    return {
        "arg": "normalization",
        "desc": "Normalization:",
        "widget": widgets.Dropdown(
            options=["None", "association", "inclusion", "jaccard", "salton"],
            layout=Layout(width="55%"),
        ),
    }


def nx_layout():
    return {
        "arg": "layout",
        "desc": "Layout:",
        "widget": widgets.Dropdown(
            options=[
                "Circular",
                "Kamada Kawai",
                "Planar",
                "Random",
                "Spectral",
                "Spring",
                "Shell",
            ],
            layout=Layout(width="55%"),
        ),
    }


def top_n(m=10, n=51, i=5):
    return {
        "arg": "top_n",
        "desc": "Top N:",
        "widget": widgets.Dropdown(
            options=list(range(m, n, i)), layout=Layout(width="55%"),
        ),
    }


def x_axis():
    return {
        "arg": "x_axis",
        "desc": "X-axis:",
        "widget": widgets.Dropdown(options=[0], layout=Layout(width="55%"),),
    }


def y_axis():
    return {
        "arg": "y_axis",
        "desc": "Y-axis:",
        "widget": widgets.Dropdown(options=[0], layout=Layout(width="55%"),),
    }


def APP(app_title, tab_titles, tab_widgets, tab=None):
    """Jupyter Lab dashboard.
    """

    if tab is not None:
        return AppLayout(
            header=widgets.HTML(
                value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                    app_title + " / " + tab_titles[tab]
                )
            ),
            center=tab_widgets[tab],
            pane_heights=["80px", "660px", 0],  # tamaño total de la ventana: Ok!
        )

    body = widgets.Tab()
    body.children = tab_widgets
    for i in range(len(tab_widgets)):
        body.set_title(i, tab_titles[i])
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                app_title
            )
        ),
        center=body,
        pane_heights=["80px", "720px", 0],
    )


def TABapp(left_panel, server, output):

    #  defines interactive output
    args = {control["arg"]: control["widget"] for control in left_panel}
    with output:
        display(widgets.interactive_output(server, args,))

    grid = GridspecLayout(13, 6, height="650px")

    # left panel layout
    grid[0:, 0] = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.Label(value=left_panel[index]["desc"]),
                    left_panel[index]["widget"],
                ],
                layout=Layout(
                    display="flex", justify_content="flex-end", align_content="center",
                ),
            )
            for index in range(len(left_panel))
        ]
    )

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

    # output
    grid[:, 1:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )

    return grid


class TABapp_:
    def __init__(self):
        self._output = widgets.Output()
        self._panel = []
        self._grid = []
        self._result = None
        self._cmap = None
        self._view = None

    def run(self):
        return self._grid

    def gui(self):
        pass

    def update(self, button):
        pass

    # def update(self, button):
    #     self.compute()
    #     self.display()

    # def app_output(self):

    #     self._output.clear_output()

    #     with self._output:

    #         if self._plot == "*Matrix*":
    #             display(matrix.style.background_gradient(cmap=self._cmap, axis=bg_axis))
    #             return

    #         if self._plot == "Heatmap":
    #             display(
    #                 plt.heatmap(self._result, cmap=self._cmap, figsize=self._figsize)
    #             )
    #             return

    #         if self._plot == "Bubble plot":
    #             display(
    #                 plt.bubble(self._result, cmap=self._cmap, figsize=self._figsize)
    #             )
    #             return

    #         if self._plot == "Slope chart":
    #             display(
    #                 slope_chart(
    #                     self._result,
    #                     figsize=self._figsize,
    #                     cmap_column=self._cmap,
    #                     cmap_by=self._cmap_by,
    #                 )
    #             )
    #             return

    #         if self._plot == "Bar plot":
    #             return

    #         if self._plot == "Horizontal bar plot":
    #             return

    #         if self._plot == "Pie":
    #             return

    #         if self._plot == "Gant":
    #             return

    def create_grid(self):

        #  Grid size
        self._grid = GridspecLayout(14, 6, height="650px")

        # Button control
        self._grid[0, 0] = widgets.Button(
            description="Update", layout=Layout(width="91%", border="2px solid gray"),
        )
        self._grid[0, 0].on_click(self.update)

        self._grid[1:, 0] = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.Label(value=self._panel[index]["desc"]),
                        self._panel[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self._panel))
            ]
        )

        #  Output area
        self._grid[:, 1:] = widgets.VBox(
            [self._output], layout=Layout(height="650px", border="2px solid gray")
        )

        args = {control["arg"]: control["widget"] for control in self._panel}
        with self._output:
            display(widgets.interactive_output(self.gui, args,))
