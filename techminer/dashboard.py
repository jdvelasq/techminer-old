import ipywidgets as widgets
from IPython.display import display
from ipywidgets import GridspecLayout, Layout
import pandas as pd
from techminer.plots import COLORMAPS


#
# Common controls GUI definition
#
def affinity():
    return {
        "arg": "affinity",
        "desc": "Affinity:",
        "widget": widgets.Dropdown(
            options=["euclidean", "l1", "l2", "manhattan", "cosine"],
            layout=Layout(width="55%"),
        ),
    }


def ascending():
    return {
        "arg": "ascending",
        "desc": "Ascending:",
        "widget": widgets.Dropdown(options=[True, False], layout=Layout(width="55%"),),
    }


def c_axis_ascending():
    return {
        "arg": "c_axis_ascending",
        "desc": "C-axis ascending:",
        "widget": widgets.Dropdown(options=[True, False], layout=Layout(width="55%"),),
    }


def r_axis_ascending():
    return {
        "arg": "r_axis_ascending",
        "desc": "R-axis ascending:",
        "widget": widgets.Dropdown(options=[True, False], layout=Layout(width="55%"),),
    }


def clustering_method():
    return {
        "arg": "clustering_method",
        "desc": "Clustering Method:",
        "widget": widgets.Dropdown(
            options=[
                "Affinity Propagation",
                "Agglomerative Clustering",
                "Birch",
                "DBSCAN",
                "KMeans",
                "Mean Shift",
            ],
            layout=Layout(width="55%"),
        ),
    }


def cmap(arg="cmap", desc="Colormap:"):
    return {
        "arg": arg,
        "desc": desc,
        "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
    }


def color_scheme():
    return {
        "arg": "color_scheme",
        "desc": "Color Scheme:",
        "widget": widgets.Dropdown(
            options=[
                "4 Quadrants",
                "Clusters",
                "Greys",
                "Purples",
                "Blues",
                "Greens",
                "Oranges",
                "Reds",
            ],
            layout=Layout(width="55%"),
        ),
    }


def decomposition_method():
    return {
        "arg": "decomposition_method",
        "desc": "Decompostion method:",
        "widget": widgets.Dropdown(
            options=["Factor Analysis", "PCA", "Fast ICA", "SVD"],
            layout=Layout(width="55%"),
        ),
    }


def dropdown(desc, options):
    arg = desc.replace(":", "").replace(" ", "_").replace("-", "_").lower()
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
            options=range(5, 26, 1), layout=Layout(width="55%"),
        ),
    }


def fig_width():
    return {
        "arg": "width",
        "desc": "Width:",
        "widget": widgets.Dropdown(
            options=range(5, 26, 1), layout=Layout(width="55%"),
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


def max_iter():
    return {
        "arg": "max_iter",
        "desc": "Max iterations:",
        "widget": widgets.Dropdown(
            options=list(range(50, 501, 50)), layout=Layout(width="55%"),
        ),
    }


def max_items():
    return {
        "arg": "max_items",
        "desc": "Max items:",
        "widget": widgets.Dropdown(
            options=list(range(5, 100, 5)) + list(range(100, 3001, 100)),
            layout=Layout(width="55%"),
        ),
    }


def n_labels():
    return {
        "arg": "n_labels",
        "desc": "N labels:",
        "widget": widgets.Dropdown(
            options=list(range(10, 151, 10)), layout=Layout(width="55%"),
        ),
    }


def min_occurrence():
    return {
        "arg": "min_occurrence",
        "desc": "Min occurrence:",
        "widget": widgets.Dropdown(
            options=list(range(1, 21)), layout=Layout(width="55%"),
        ),
    }


def nx_iterations():
    return {
        "arg": "nx_iterations",
        "desc": "nx iterations:",
        "widget": widgets.Dropdown(
            options=list(range(5, 101, 1)), layout=Layout(width="55%"),
        ),
    }


def n_iter():
    return {
        "arg": "n_iter",
        "desc": "Iterations:",
        "widget": widgets.Dropdown(
            options=list(range(5, 51, 1)), layout=Layout(width="55%"),
        ),
    }


def n_clusters(m=3, n=21, i=1):
    return {
        "arg": "n_clusters",
        "desc": "N Clusters:",
        "widget": widgets.Dropdown(
            options=list(range(m, n, i)), layout=Layout(width="55%"),
        ),
    }


def n_clusters_ac():
    return {
        "arg": "n_clusters",
        "desc": "N Clusters:",
        "widget": widgets.Dropdown(
            options=["None"] + list(range(2, 21)), layout=Layout(width="55%"),
        ),
    }


def n_components():
    return {
        "arg": "n_components",
        "desc": "N components:",
        "widget": widgets.Dropdown(
            options=list(range(2, 11)), layout=Layout(width="55%"),
        ),
    }


def normalization(include_none=True):
    options = sorted(
        [
            "Association",
            "Jaccard",
            "Dice",
            "Salton/Cosine",
            "Equivalence",
            "Inclusion",
            "Mutual Information",
        ]
    )
    if include_none is True:
        options = ["None"] + options
    return {
        "arg": "normalization",
        "desc": "Normalization:",
        "widget": widgets.Dropdown(options=options, layout=Layout(width="55%"),),
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


def random_state():
    return {
        "arg": "random_state",
        "desc": "Random State:",
        "widget": widgets.Dropdown(
            options=[
                "0012345",
                "0123456",
                "0234567",
                "0345678",
                "0456789",
                "0567890",
                "0678901",
                "0789012",
                "0890123",
                "0901234",
                "1012345",
                "1123456",
                "1234567",
                "1345678",
                "1456789",
                "1567890",
                "1678901",
                "1789012",
                "1890123",
                "1901234",
                "2012345",
                "2123456",
                "2234567",
                "2345678",
                "2456789",
                "2567890",
                "2678901",
                "2789012",
                "2890123",
                "2901234",
                "3012345",
            ],
            layout=Layout(width="55%"),
        ),
    }


def separator(text):
    return {
        "desc": "*SEPARATOR*",
        "widget": widgets.HTML("<b>" + text + "</b><hr>"),
    }


def top_n(m=10, n=51, i=5):
    return {
        "arg": "top_n",
        "desc": "Top N:",
        "widget": widgets.Dropdown(
            options=list(range(m, n, i)), layout=Layout(width="55%"),
        ),
    }


def x_axis(n=10):
    return {
        "arg": "x_axis",
        "desc": "X-axis:",
        "widget": widgets.Dropdown(options=list(range(n)), layout=Layout(width="55%"),),
    }


def y_axis(n=10, value=1):
    return {
        "arg": "y_axis",
        "desc": "Y-axis:",
        "widget": widgets.Dropdown(
            options=list(range(n)), value=1, layout=Layout(width="55%"),
        ),
    }


def processing():
    html = """
        <style>
        .loader {
        border: 16px solid #f3f3f3;
        border-radius: 50%;
        border-top: 16px solid #3498db;
        width: 70px;
        height: 70px;
        -webkit-animation: spin 2s linear infinite; /* Safari */
        animation: spin 2s linear infinite;
        }

        /* Safari */
        @-webkit-keyframes spin {
        0% { -webkit-transform: rotate(0deg); }
        100% { -webkit-transform: rotate(360deg); }
        }

        @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
        }
        </style>
        </head>
        <h3>Processing ... </h3>
        <div class="loader"></div>        

        """
    return widgets.HTML(html)


class DASH:
    def __init__(self):

        ## layout
        self.app_layout = []

        ## display pandas options
        self.pandas_max_rows = 50
        self.pandas_max_columns = 100

        ## Panel controls
        self.app_title = "None"
        self.menu_options = None
        self.menu = None
        self.panel_widgets = []
        self.output = None

    def run(self):
        return self.app_layout

    def interactive_output(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def calculate(self, button):

        menu = (
            self.menu.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .lower()
        )
        self.output.clear_output()
        with self.output:
            print("Processing ...")
        result = getattr(self, menu)()
        self.output.clear_output()
        with pd.option_context(
            "max_rows", self.pandas_max_rows, "max_columns", self.pandas_max_columns
        ):
            with self.output:
                display(result)

    def create_grid(self):

        ## Grid size
        panel_len = 0
        if self.panel_widgets is not None:
            panel_len += len(self.panel_widgets)
        self.app_layout = GridspecLayout(max(14, panel_len + 1), 4, height="770px")

        ## Panel Title
        panel_widgets = [
            widgets.HTML(
                "<h2>"
                + self.app_title
                + "</h2>"
                + "<hr style='height:3px;border-width:0;color:gray;background-color:black'>"
            )
        ]

        # menu
        if self.menu_options is not None:
            self.menu_widget = widgets.Dropdown(
                options=self.menu_options, layout=Layout(width="98%"),
            )
            panel_widgets += [self.menu_widget]
        else:
            self.menu_widget = None

        ## Update controls
        if self.panel_widgets is not None:
            for index, _ in enumerate(self.panel_widgets):
                if self.panel_widgets[index]["desc"] == "*SEPARATOR*":
                    panel_widgets.append(self.panel_widgets[index]["widget"])
                else:
                    panel_widgets.append(
                        widgets.HBox(
                            [
                                widgets.Label(value=self.panel_widgets[index]["desc"]),
                                self.panel_widgets[index]["widget"],
                            ],
                            layout=Layout(
                                display="flex",
                                justify_content="flex-end",
                                align_content="center",
                            ),
                        )
                    )

        ## Calculate button
        if self.menu_options is not None:
            calculate_button = widgets.Button(
                description="Apply",
                layout=Layout(width="98%", border="2px solid gray"),
            )
            calculate_button.on_click(self.calculate)
            panel_widgets += [calculate_button]

        ## Build left panel
        self.app_layout[:, 0] = widgets.VBox(panel_widgets)

        ## Output area
        self.output = widgets.Output()
        self.app_layout[:, 1:] = widgets.VBox(
            [self.output], layout=Layout(height="770px", border="2px solid gray"),
        )

        ## interactive
        args = {}
        if self.menu_options is not None:
            args = {
                **args,
                **{"menu": self.menu_widget},
            }
        if self.panel_widgets is not None:
            args = {
                **args,
                **{
                    control["arg"]: control["widget"]
                    for control in self.panel_widgets
                    if control["desc"] != "*SEPARATOR*"
                },
            }

        widgets.interactive_output(
            self.interactive_output, args,
        )

    def text_transform(self, text):
        return (
            text.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "")
            .lower()
        )

    def set_enabled(self, name):
        for index, _ in enumerate(self.panel_widgets):
            x = self.text_transform(self.panel_widgets[index]["desc"])
            name = self.text_transform(name)
            if x == name:
                self.panel_widgets[index]["widget"].disabled = False
                return

    def set_disabled(self, name):
        for index, _ in enumerate(self.panel_widgets):
            x = self.text_transform(self.panel_widgets[index]["desc"])
            name = self.text_transform(name)
            if x == name:
                self.panel_widgets[index]["widget"].disabled = True
                return

    def set_options(self, name, options):
        for index, _ in enumerate(self.panel_widgets):
            x = self.text_transform(self.panel_widgets[index]["desc"])
            name = self.text_transform(name)
            if x == name:
                self.panel_widgets[index]["widget"].options = options
                return

    def enable_disable_clustering_options(self, include_random_state=False):

        if self.clustering_method in ["Affinity Propagation"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_enabled("Random State:")

        if self.clustering_method in ["Agglomerative Clustering"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["Birch"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["DBSCAN"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["Feature Agglomeration"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["KMeans"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["Mean Shift"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

