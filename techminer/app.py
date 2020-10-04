import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
import pandas as pd


header_style = """
<style>
.hbox_style{
    width:99.9%;
    border : 2px solid #ff8000;
    height: auto;
    background-color:#ff8000;
    box-shadow: 2px 2px lightgray;
}

.widget-select > select {background-color: #ff8000; color: white; border-color: light-gray;}

</style>
"""


class App:
    def __init__(self):

        ## layout
        self.app_layout = []
        self.apps = [
            "Column Explorer",
            "Matrix Explorer",
        ]

        self.app_layout = GridspecLayout(14, 4, height="800px")

        #
        app_menu_widget = widgets.Dropdown(
            options=self.apps,
            layout=Layout(
                width="70%",
            ),
        )

        self.app_layout[0, :] = widgets.HBox(
            [
                widgets.HTML(header_style),
                widgets.HTML('<h2 style="color:white;">TechMiner</h2>'),
                widgets.Dropdown(
                    options=self.apps,
                    layout=Layout(width="70%"),
                ).add_class("widget-select"),
            ],
            layout=Layout(
                display="flex",
                justify_content="space-between",
                align_items="center",
            ),
        ).add_class("hbox_style")

    def run(self):
        return self.app_layout
