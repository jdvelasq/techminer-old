import ipywidgets as widgets
from ipywidgets import AppLayout, GridspecLayout, Layout


class DASH:
    def __init__(self):

        ## layout
        self.grid_ = []
        self.tab_ = None

        ## Panel controls
        self.app_title_ = "None"
        self.menu_options_ = None
        self.main_panel_ = []
        self.aux_panel_ = []
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
        if self.main_panel_ is not None:
            panel_len += len(self.main_panel_)
        if self.aux_panel_ is not None:
            panel_len += len(self.aux_panel_)
        self.grid_ = GridspecLayout(max(14, panel_len + 1), 5, height="720px")

        ## Menu
        self.menu_ = widgets.Dropdown(
            options=self.menu_options_, layout=Layout(width="98%"),
        )

        ## Buttons
        if self.main_panel_ is not None:
            calculate_button = widgets.Button(
                description="Calculate",
                layout=Layout(width="100%", border="2px solid gray"),
            )
            calculate_button.on_click(self.calculate)

        update_button = widgets.Button(
            description="Update Visualization",
            layout=Layout(width="98%", border="2px solid gray"),
        )
        update_button.on_click(self.update)

        ## Left panel layout
        panel_layout = [
            widgets.HTML(
                "<h2>"
                + self.app_title_
                + "</h2>"
                + "<hr style='height:2px;border-width:0;color:gray;background-color:gray'>"
            )
        ]
        panel_layout += [self.menu_]
        if self.main_panel_ is not None:
            panel_layout += [calculate_button]
            panel_layout += [
                widgets.HBox(
                    [
                        widgets.Label(value=self.main_panel_[index]["desc"]),
                        self.main_panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self.main_panel_))
            ]

        if self.aux_panel_ is not None:
            panel_layout += [update_button]
            panel_layout += [
                widgets.HBox(
                    [
                        widgets.Label(value=self.aux_panel_[index]["desc"]),
                        self.aux_panel_[index]["widget"],
                    ],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-end",
                        align_content="center",
                    ),
                )
                for index in range(len(self.aux_panel_))
            ]

        self.grid_[:, 0] = widgets.VBox(panel_layout)

        ## Output tabs
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
        args = {"menu": self.menu_}
        if self.main_panel_ is not None:
            args = {
                **args,
                **{control["arg"]: control["widget"] for control in self.main_panel_},
            }
        if self.aux_panel_ is not None:
            args = {
                **args,
                **{control["arg"]: control["widget"] for control in self.aux_panel_},
            }

        #  with self.output_:
        widgets.interactive_output(
            self.gui, args,
        )

        ## self.tab_.observe(self.on_selected_tab_changes, names="selected_index")
