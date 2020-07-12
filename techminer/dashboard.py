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
        # self.TAB_menu_options_ = None
        self.menu_options_ = None
        self.panel_ = []

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
        if self.panel_ is not None:
            panel_len += len(self.panel_)
        self.grid_ = GridspecLayout(max(14, panel_len + 1), 5, height="720px")

        ## Panel Title
        panel_layout = [
            widgets.HTML(
                "<h2>"
                + self.app_title_
                + "</h2>"
                + "<hr style='height:3px;border-width:0;color:gray;background-color:black'>"
            )
        ]

        # if self.TAB_menu_options_ is not None:
        #     self.TAB_menu_ = widgets.Dropdown(
        #         options=self.TAB_menu_options_, layout=Layout(width="98%"),
        #     )
        #     panel_layout += [self.TAB_menu_]
        # else:
        #     self.TAB_menu_ = None

        # ## line
        # if self.calculate_panel_ is not None and self.update_panel_ is not None:
        #     panel_layout += [
        #         widgets.HTML(
        #             "<hr style='height:3px;border-width:0;color:gray;background-color:black'>"
        #         )
        #     ]

        ## menu
        if self.menu_options_ is not None:
            self.menu_ = widgets.Dropdown(
                options=self.menu_options_, layout=Layout(width="98%"),
            )
            panel_layout += [self.menu_]
        else:
            self.menu_ = None

        ## Update controls
        if self.panel_ is not None:
            panel_layout += [
                widgets.HBox(
                    [
                        widgets.Label(value=self.panel_[index]["desc"]),
                        self.panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self.panel_))
            ]

        # Calculate button
        if self.menu_options_ is not None:
            calculate_button = widgets.Button(
                description="Calculate",
                layout=Layout(width="98%", border="2px solid gray"),
            )
            calculate_button.on_click(self.update)
            panel_layout += [calculate_button]

        ## Build left panel
        self.grid_[:, 0] = widgets.VBox(panel_layout)

        ## Output area
        self.output_ = widgets.Output()
        self.grid_[:, 1:] = widgets.VBox(
            [self.output_], layout=Layout(height="720px", border="2px solid gray"),
        )

        ## interactive
        args = {}
        if self.menu_ is not None:
            args = {
                **args,
                **{"menu": self.menu_},
            }
        if self.panel_ is not None:
            args = {
                **args,
                **{control["arg"]: control["widget"] for control in self.panel_},
            }
        # if self.update_menu_ is not None:
        #     args = {
        #         **args,
        #         **{"update_menu": self.update_menu_},
        #     }
        # if self.update_panel_ is not None:
        #     args = {
        #         **args,
        #         **{control["arg"]: control["widget"] for control in self.update_panel_},
        #     }

        widgets.interactive_output(
            self.gui, args,
        )

        ## self.tab_.observe(self.on_selected_tab_changes, names="selected_index")
