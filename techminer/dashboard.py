import ipywidgets as widgets
from ipywidgets import AppLayout, GridspecLayout, Layout


class DASH:
    def __init__(self):

        ##
        self._obj = None

        ## layout
        self.grid_ = []
        self.tab_ = None

        ## Panel controls
        self.app_title_ = "None"
        self.calculate_menu_options_ = None
        self.calculate_panel_ = []
        self.update_menu_options_ = None
        self.update_panel_ = []
        self.tab_titles_ = []

    def run(self):
        return self.grid_

    def gui(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def calculate(self, button):
        pass

    def update(self, button):
        pass

    def create_grid(self):

        ## Grid size
        panel_len = 0
        if self.calculate_panel_ is not None:
            panel_len += len(self.calculate_panel_)
        if self.update_panel_ is not None:
            panel_len += len(self.update_panel_)
        self.grid_ = GridspecLayout(max(14, panel_len + 1), 5, height="720px")

        ## Panel Title
        panel_layout = [
            widgets.HTML(
                "<h2>"
                + self.app_title_
                + "</h2>"
                + "<hr style='height:2px;border-width:0;color:gray;background-color:gray'>"
            )
        ]

        ## Calculate menu
        if self.calculate_menu_options_ is not None:
            self.calculate_menu_ = widgets.Dropdown(
                options=self.calculate_menu_options_, layout=Layout(width="98%"),
            )
            panel_layout += [self.calculate_menu_]
        else:
            self.calculate_menu_ = None

        ## Calculate Panel
        if self.calculate_panel_ is not None:

            # Caulculate controls
            panel_layout += [
                widgets.HBox(
                    [
                        widgets.Label(value=self.calculate_panel_[index]["desc"]),
                        self.calculate_panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self.calculate_panel_))
            ]

        # Calculate button
        if (
            self.calculate_menu_options_ is not None
            or self.calculate_panel_ is not None
        ):
            calculate_button = widgets.Button(
                description="Calculate",
                layout=Layout(width="98%", border="2px solid gray"),
            )
            calculate_button.on_click(self.calculate)
            panel_layout += [calculate_button]

        ## Update menu
        if self.update_menu_options_ is not None:
            self.update_menu_ = widgets.Dropdown(
                options=self.update_menu_options_, layout=Layout(width="98%"),
            )
            panel_layout += [self.update_menu_]
        else:
            self.update_menu_ = None

        ## Update controls
        if self.update_panel_ is not None:
            panel_layout += [
                widgets.HBox(
                    [
                        widgets.Label(value=self.update_panel_[index]["desc"]),
                        self.update_panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self.update_panel_))
            ]

        # Update button
        if self.update_menu_options_ is not None or self.update_panel_ is not None:
            update_button = widgets.Button(
                description="Update Visualization",
                layout=Layout(width="98%", border="2px solid gray"),
            )
            update_button.on_click(self.update)
            panel_layout += [update_button]

        ## Build left panel
        self.grid_[:, 0] = widgets.VBox(panel_layout)

        ## Output area
        self.output_ = widgets.Output()
        self.grid_[:, 1:] = widgets.VBox(
            [self.output_], layout=Layout(height="720px", border="2px solid gray"),
        )
        #  [widgets.Output() for _ in range(len(self.tab_titles_))]
        # self.tab_ = widgets.Tab()
        # self.tab_.children = [
        #     widgets.VBox(
        #         [self.output_[i]],
        #         layout=Layout(height="657px", border="2px solid gray"),
        #     )
        #     for i, _ in enumerate(self.tab_titles_)
        # ]
        # for i, title in enumerate(self.tab_titles_):
        #     self.tab_.set_title(i, title)
        # self.grid_[:, 1:] = self.tab_

        ## interactive
        args = {}
        if self.calculate_menu_ is not None:
            args = {
                **args,
                **{"calculate_menu": self.calculate_menu_},
            }
        if self.calculate_panel_ is not None:
            args = {
                **args,
                **{
                    control["arg"]: control["widget"]
                    for control in self.calculate_panel_
                },
            }
        if self.update_menu_ is not None:
            args = {
                **args,
                **{"update_menu": self.update_menu_},
            }
        if self.update_panel_ is not None:
            args = {
                **args,
                **{control["arg"]: control["widget"] for control in self.update_panel_},
            }

        widgets.interactive_output(
            self.gui, args,
        )

        ## self.tab_.observe(self.on_selected_tab_changes, names="selected_index")
