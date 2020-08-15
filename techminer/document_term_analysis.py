from techminer.tfidf import TF_matrix, TFIDF_matrix

import techminer.common as cmn
import techminer.dashboard as dash
from techminer.bar_plot import bar_plot
from techminer.barh_plot import barh_plot
from techminer.dashboard import DASH


###############################################################################
##
##  Model
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def apply(self):

        matrix = TF_matrix(data=self.data, column=self.column, scheme="raw")
        matrix = cmn.limit_to_exclude(
            data=matrix,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        TF_matrix_ = cmn.add_counters_to_axis(
            X=matrix, axis=1, data=self.data, column=self.column
        )
        if self.norm is not None:
            self.norm = self.norm.lower()

        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )
        vector = TFIDF_matrix_.sum(axis=0).sort_values(ascending=False).head(self.top_n)
        TFIDF_matrix_ = vector.to_frame()
        TFIDF_matrix_.columns = ["TF*IDF"]

        if self.sort_by == "TF*IDF":
            TFIDF_matrix_["TEXT"] = TFIDF_matrix_.index.map(lambda w: w.split(" ")[-1])
            TFIDF_matrix_ = TFIDF_matrix_.sort_values(
                ["TF*IDF", "TEXT"], ascending=self.ascending
            )
            TFIDF_matrix_.pop("TEXT")
        else:
            TFIDF_matrix_ = cmn.sort_by_axis(
                data=TFIDF_matrix_,
                sort_by=self.sort_by,
                ascending=self.ascending,
                axis=0,
            )

        if self.use_idf is False:
            TFIDF_matrix_["TF*IDF"] = TFIDF_matrix_["TF*IDF"].map(int)

        self.X_ = TFIDF_matrix_

    def table(self):
        self.apply()
        return self.X_.style.set_precision(2).background_gradient(cmap=self.cmap)

    def bar_plot(self):
        self.apply()
        return bar_plot(
            height=self.X_["TF*IDF"],
            darkness=None,
            cmap=self.cmap,
            figsize=(self.width, self.height),
            ylabel="TF*IDF",
        )

    def horizontal_bar_plot(self):
        self.apply()
        return barh_plot(
            width=self.X_["TF*IDF"],
            darkness=None,
            cmap=self.cmap,
            figsize=(self.width, self.height),
            xlabel="TF*IDF",
        )


###############################################################################
##
##  DASH
##
###############################################################################

COLUMNS = {
    "Author_Keywords",
    "Index_Keywords",
    "Author_Keywords_CL",
    "Index_Keywords_CL",
    "Abstract_words",
    "Title_words",
    "Abstract_words_CL",
    "Title_words_CL",
}


class DASHapp(DASH, Model):
    def __init__(self, data, limit_to=None, exclude=None, years_range=None):
        """Dashboard app"""

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.pandas_max_rows = 300

        self.app_title = "TF*IDF Analysis"
        self.menu_options = ["Table", "Bar plot", "Horizontal bar plot"]

        self.panel_widgets = [
            dash.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            dash.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
            dash.dropdown(desc="Use IDF:", options=[True, False,],),
            dash.dropdown(desc="Smooth IDF:", options=[True, False,],),
            dash.dropdown(desc="Sublinear TF:", options=[True, False,],),
            dash.separator(text="Visualization"),
            dash.top_n(m=10, n=301, i=10),
            dash.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "TF*IDF",],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == self.menu_options[0]:
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Width:")
            self.set_enabled("Height:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
    ).run()

