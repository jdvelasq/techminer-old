import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.dashboard import DASH
from techminer.explode import __explode

###############################################################################
##
##  CALCULATIONS
##
###############################################################################


def TF_matrix(data, column, scheme=None, min_occurrence=1):
    X = data[[column, "ID"]].copy()
    X["value"] = 1.0
    X = __explode(X, column)
    result = pd.pivot_table(
        data=X, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)

    terms = result.sum(axis=0)
    terms = terms.sort_values(ascending=False)
    terms = terms[terms >= min_occurrence]
    result = result.loc[:, terms.index]

    rows = result.sum(axis=1)
    rows = rows[rows > 0]
    result = result.loc[rows.index, :]

    if scheme is None or scheme == "raw":
        return result

    if scheme == "binary":
        result = result.applymap(lambda w: 1 if w > 0 else 0)

    if scheme == "log":
        result = result.applymap(lambda w: np.log(1 + w))

    return result


def TFIDF_matrix(
    TF_matrix,
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False,
    max_items=3000,
):

    result = (
        TfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )
        .fit_transform(TF_matrix)
        .toarray()
    )

    result = pd.DataFrame(result, columns=TF_matrix.columns)

    if len(result.columns) > max_items:
        terms = result.sum(axis=0)
        terms = terms.sort_values(ascending=False)
        terms = terms.head(max_items)
        result = result.loc[:, terms.index]
        rows = result.sum(axis=1)
        rows = rows[rows > 0]
        result = result.loc[rows.index, :]

    return result


###############################################################################
##
##  Model
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude):
        #
        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude
        ##
        self.ascending = None
        self.sort_by = None
        self.norm = None
        self.column = None
        self.smooth_idf = None
        self.use_idf = None
        self.sublinear_tf = None
        self.top_n = None
        self.cmap = None
        self.height = None
        self.width = None
        ##

    def fit(self):

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
            TFIDF_matrix_ = TFIDF_matrix_.sort_values(
                "TF*IDF", ascending=self.ascending
            )
        else:
            TFIDF_matrix_ = cmn.sort_by_axis(
                data=TFIDF_matrix_,
                sort_by=self.sort_by,
                ascending=self.ascending,
                axis=0,
            )

        self.X_ = TFIDF_matrix_

    def table(self):
        self.fit()
        return self.X_.style.background_gradient(cmap=self.cmap)

    def bar_plot(self):
        self.fit()
        return plt.bar(
            height=self.X_["TF*IDF"],
            darkness=None,
            cmap=self.cmap,
            figsize=(self.width, self.height),
            ylabel="TF*IDF",
        )

    def horizontal_bar_plot(self):
        self.fit()
        return plt.barh(
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
    def __init__(self, data, limit_to=None, exclude=None):
        """Dashboard app"""

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.data = data
        self.app_title = "TF*IDF Analysis"
        self.menu_options = ["Table", "Bar plot", "Horizontal bar plot"]

        self.panel_widgets = [
            dash.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            dash.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
            dash.dropdown(desc="Use IDF:", options=[True, False,],),
            dash.dropdown(desc="Smooth IDF:", options=[True, False,],),
            dash.dropdown(desc="Sublinear TF:", options=[True, False,],),
            dash.separator(text="Visualization"),
            dash.top_n(),
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


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()
