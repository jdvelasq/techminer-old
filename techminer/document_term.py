import numpy as np
import pandas as pd
from techminer.explode import __explode
from sklearn.feature_extraction.text import TfidfTransformer
from IPython.display import clear_output, display
import techminer.gui as gui
import techminer.plots as plt
import techminer.common as cmn
from techminer.dashboard import DASH

###############################################################################
##
##  CALCULATIONS
##
###############################################################################


def TF_matrix(data, column, scheme=None):
    X = data[[column, "ID"]].copy()
    X["value"] = 1.0
    X = __explode(X, column)
    result = pd.pivot_table(
        data=X, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)

    if scheme is None or scheme == "raw":
        return result

    if scheme == "binary":
        result = result.applymap(lambda w: 1 if w > 0 else 0)

    if scheme == "log":
        result = result.applymap(lambda w: np.log(1 + w))

    return result


def TFIDF_matrix(
    TF_matrix, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False
):

    result = (
        TfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )
        .fit_transform(TF_matrix)
        .toarray()
    )

    result = pd.DataFrame(result, columns=TF_matrix.columns)
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
            gui.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            gui.top_n(),
            gui.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
            gui.dropdown(desc="Use IDF:", options=[True, False,],),
            gui.dropdown(desc="Smooth IDF:", options=[True, False,],),
            gui.dropdown(desc="Sublinear TF:", options=[True, False,],),
            gui.dropdown(
                desc="Sort by:",
                options=["Alphabetic", "Num Documents", "Times Cited", "TF*IDF",],
            ),
            gui.ascending(),
            gui.cmap(),
            gui.fig_width(),
            gui.fig_height(),
        ]
        super().create_grid()

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        self.panel_widgets[-2]["widget"].disabled = self.menu == self.menu_options[0]
        self.panel_widgets[-1]["widget"].disabled = self.menu == self.menu_options[0]


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(data, limit_to=None, exclude=None):
    return DASHapp(data=data, limit_to=limit_to, exclude=exclude).run()


###############################################################################
##
##  TAB app
##
###############################################################################


# class TABapp0(gui.TABapp_):
#     """
#     """

#     def __init__(self, data, limit_to, exclude):

#         super(TABapp0, self).__init__()

#         self.data_ = data
#         self.limit_to_ = limit_to
#         self.exclude_ = exclude

#         self.panel_ = [
#             gui.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
#             gui.top_n(),
#             gui.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
#             gui.dropdown(desc="Use IDF:", options=[True, False,],),
#             gui.dropdown(desc="Smooth IDF:", options=[True, False,],),
#             gui.dropdown(desc="Sublinear TF:", options=[True, False,],),
#             gui.dropdown(
#                 desc="View:", options=["Table", "Bar plot", "Horizontal bar plot",],
#             ),
#             gui.dropdown(
#                 desc="Sort by:",
#                 options=["Alphabetic", "Num Documents", "Times Cited", "TF*IDF",],
#             ),
#             gui.ascending(),
#             gui.cmap(),
#             gui.fig_width(),
#             gui.fig_height(),
#         ]
#         super().create_grid()

#     def gui(self, **kwargs):

#         super().gui(**kwargs)
#         self.panel_[10]["widget"].disabled = kwargs["view"] == "Table"
#         self.panel_[11]["widget"].disabled = kwargs["view"] == "Table"

#     def update(self, button):
#         """
#         """

#         figsize = (self.width, self.height)

#         matrix = TF_matrix(data=self.data_, column=self.column)
#         matrix = cmn.limit_to_exclude(
#             data=matrix,
#             axis=1,
#             column=self.column,
#             limit_to=self.limit_to_,
#             exclude=self.exclude_,
#         )
#         TF_matrix_ = cmn.add_counters_to_axis(
#             X=matrix, axis=1, data=self.data_, column=self.column
#         )

#         if self.norm is not None:
#             self.norm = self.norm.lower()
#         TFIDF_matrix_ = TFIDF_matrix(
#             TF_matrix=TF_matrix_,
#             norm=self.norm,
#             use_idf=self.use_idf,
#             smooth_idf=self.smooth_idf,
#             sublinear_tf=self.sublinear_tf,
#         )
#         vector = TFIDF_matrix_.sum(axis=0).sort_values(ascending=False).head(self.top_n)
#         TFIDF_matrix_ = vector.to_frame()
#         TFIDF_matrix_.columns = ["TF*IDF"]

#         if self.sort_by == "TF*IDF":
#             TFIDF_matrix_ = TFIDF_matrix_.sort_values(
#                 "TF*IDF", ascending=self.ascending
#             )
#         else:
#             TFIDF_matrix_ = cmn.sort_by_axis(
#                 data=TFIDF_matrix_,
#                 sort_by=self.sort_by,
#                 ascending=self.ascending,
#                 axis=0,
#             )

#         figsize = (self.width, self.height)

#         self.output_.clear_output()
#         with self.output_:

#             if self.view == "Table":

#                 display(TFIDF_matrix_.style.background_gradient(cmap=self.cmap))

#             if self.view == "Bar plot":

#                 display(
#                     plt.bar(
#                         height=TFIDF_matrix_["TF*IDF"],
#                         darkness=None,
#                         cmap=self.cmap,
#                         figsize=figsize,
#                         ylabel="TF*IDF",
#                     )
#                 )

#             if self.view == "Horizontal bar plot":

#                 display(
#                     plt.barh(
#                         width=TFIDF_matrix_["TF*IDF"],
#                         darkness=None,
#                         cmap=self.cmap,
#                         figsize=figsize,
#                         xlabel="TF*IDF",
#                     )
#                 )


# def document_term_matrix(data, column):
#     """Computes the document-term matrix for the terms in a column.

#     Args:
#         column (str): the column to explode.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
#     >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
#     >>> df = pd.DataFrame(
#     ...    {
#     ...       'Authors': x,
#     ...       'Author_Keywords': y,
#     ...       "Times_Cited": list(range(len(x))),
#     ...       'ID': list(range(len(x))),
#     ...    }
#     ... )
#     >>> df
#       Authors Author_Keywords  Times_Cited  ID
#     0       A               a            0   0
#     1     A;B             a;b            1   1
#     2       B               b            2   2
#     3   A;B;C               c            3   3
#     4     B;D             c;d            4   4

#     >>> TF_matrix(df, 'Authors')
#        A  B  C  D
#     0  1  0  0  0
#     1  1  1  0  0
#     2  0  1  0  0
#     3  1  1  1  0
#     4  0  1  0  1

#     >>> TF_matrix(df, 'Author_Keywords')
#        a  b  c  d
#     0  1  0  0  0
#     1  1  1  0  0
#     2  0  1  0  0
#     3  0  0  1  0
#     4  0  0  1  1

#     """
#     X = data[[column, "ID"]].copy()
#     X["value"] = 1.0
#     X = __explode(X, column)
#     result = pd.pivot_table(
#         data=X, index="ID", columns=column, margins=False, fill_value=0.0,
#     )
#     result.columns = [b for _, b in result.columns]
#     result = result.reset_index(drop=True)
#     return result

