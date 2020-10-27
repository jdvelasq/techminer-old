import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
import pandas as pd

from techminer.column_explorer import column_explorer


header_style = """
<style>
.hbox_style{
    width:99.9%;
    border : 2px solid #ff8000;
    height: auto;
    background-color:#ff8000;
    box-shadow: 1px 5px  4px #BDC3C7;
}

.app{
    background-color:#F4F6F7;
}

.output_color{
    background-color:#FFFFFF;
}

.select > select {background-color: #ff8000; color: white; border-color: light-gray;}

</style>
"""

APPS = {
    "Column Explorer": column_explorer,
}


class App:
    def __init__(self):

        #
        # APPs menu
        #
        apps_dropdown = widgets.Dropdown(
            options=[key for key in sorted(APPS.keys())],
            layout=Layout(width="70%"),
        ).add_class("select")

        #
        # Grid layout definition
        #
        self.app_layout = GridspecLayout(
            11,
            4,
            height="780px",  # layout=Layout(border="1px solid #ff8000")
        ).add_class("app")

        #
        # Populates the grid
        #
        self.app_layout[0, :] = widgets.HBox(
            [
                widgets.HTML(header_style),
                widgets.HTML('<h2 style="color:white;">TechMiner</h2>'),
                apps_dropdown,
            ],
            layout=Layout(
                display="flex",
                justify_content="space-between",
                align_items="center",
            ),
        ).add_class("hbox_style")

        #
        # Interative output
        #
        widgets.interactive_output(
            self.interactive_output,
            {"selected-app": apps_dropdown},
        )

    def run(self):
        return self.app_layout

    def interactive_output(self, **kwargs):
        self.app_layout[1:, :] = APPS[kwargs["selected-app"]]()
