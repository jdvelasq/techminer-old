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
