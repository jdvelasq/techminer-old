import ipywidgets as widgets
from ipywidgets import AppLayout, GridspecLayout, Layout
from IPython.display import display


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

        menu = self.menu.replace(" ", "_").replace("-", "_").lower()
        self.output.clear_output()
        with self.output:
            display(getattr(self, menu)())

    def create_grid(self):

        ## Grid size
        panel_len = 0
        if self.panel_widgets is not None:
            panel_len += len(self.panel_widgets)
        self.app_layout = GridspecLayout(max(14, panel_len + 1), 5, height="720px")

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
            panel_widgets += [
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
                for index, _ in enumerate(self.panel_widgets)
            ]

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
                **{control["arg"]: control["widget"] for control in self.panel_widgets},
            }

        widgets.interactive_output(
            self.interactive_output, args,
        )
