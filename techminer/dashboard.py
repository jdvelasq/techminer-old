import ipywidgets as widgets
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.plots import COLORMAPS


#
# Common controls GUI definition
#
def affinity():
    return {
        "arg": "affinity",
        "desc": "Affinity:",
        "widget": widgets.Dropdown(
            options=["euclidean", "l1", "l2", "manhattan", "cossine"],
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


def cmap(arg="cmap", desc="Colormap:"):
    return {
        "arg": arg,
        "desc": desc,
        "widget": widgets.Dropdown(options=COLORMAPS, layout=Layout(width="55%"),),
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
            options=list(range(100, 3001, 100)), layout=Layout(width="55%"),
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
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], layout=Layout(width="55%"),
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


def n_clusters():
    return {
        "arg": "n_clusters",
        "desc": "N Clusters:",
        "widget": widgets.Dropdown(
            options=list(range(2, 21)), layout=Layout(width="55%"),
        ),
    }


def n_components():
    return {
        "arg": "n_components",
        "desc": "N components:",
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
        with self.output:
            display(result)

    def create_grid(self):

        ## Grid size
        panel_len = 0
        if self.panel_widgets is not None:
            panel_len += len(self.panel_widgets)
        self.app_layout = GridspecLayout(max(14, panel_len + 1), 4, height="720px")

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

            # panel_widgets += [
            #     widgets.HBox(
            #         [
            #             widgets.Label(value=self.panel_widgets[index]["desc"]),
            #             self.panel_widgets[index]["widget"],
            #         ],
            #         layout=Layout(
            #             display="flex",
            #             justify_content="flex-end",
            #             align_content="center",
            #         ),
            #     )
            #     for index, _ in enumerate(self.panel_widgets)
            # ]

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
            [self.output], layout=Layout(height="720px", border="2px solid gray"),
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
