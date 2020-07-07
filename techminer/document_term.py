import numpy as np
import pandas as pd
from techminer.explode import __explode
from sklearn.feature_extraction.text import TfidfTransformer
from IPython.display import clear_output, display
import techminer.gui as gui
import techminer.plots as plt
import techminer.common as cmn


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


def document_term_matrix(data, column):
    """Computes the document-term matrix for the terms in a column.

    Args:
        column (str): the column to explode.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D']
    >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
    >>> df = pd.DataFrame(
    ...    {
    ...       'Authors': x,
    ...       'Author_Keywords': y,
    ...       "Times_Cited": list(range(len(x))),
    ...       'ID': list(range(len(x))),
    ...    }
    ... )
    >>> df
      Authors Author_Keywords  Times_Cited  ID
    0       A               a            0   0
    1     A;B             a;b            1   1
    2       B               b            2   2
    3   A;B;C               c            3   3
    4     B;D             c;d            4   4

    >>> document_term_matrix(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1

    >>> document_term_matrix(df, 'Author_Keywords')
       a  b  c  d
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  0  0  1  0
    4  0  0  1  1

    """
    X = data[[column, "ID"]].copy()
    X["value"] = 1.0
    X = __explode(X, column)
    result = pd.pivot_table(
        data=X, index="ID", columns=column, margins=False, fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)
    return result


###############################################################################
##
##  APP
##
###############################################################################


def app(data, tab=None, limit_to=None, exclude=None):

    return gui.APP(
        app_title="TF-IDF",
        tab_titles=["TF-IDF"],
        tab_widgets=[TABapp0(data, limit_to, exclude).run(),],
        tab=tab,
    )


###############################################################################
##
##  TAB app
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


class TABapp0(gui.TABapp_):
    def __init__(self, data, limit_to, exclude):

        super(TABapp0, self).__init__()

        self._data = data
        self._limit_to = limit_to
        self._exclude = exclude

        self._column = None
        self._norm = None
        self._use_idf = None
        self._smooth_idf = None
        self._sublinear_tf = None
        self._top_n = None
        self._sort_by = None
        self._ascending = None

        self._TF_matrix = None
        self._TFIDF_vector_50 = None

        self._panel = [
            gui.dropdown(desc="Column:", options=[t for t in data if t in COLUMNS],),
            gui.top_n(),
            gui.dropdown(desc="Norm:", options=[None, "L1", "L2"],),
            gui.dropdown(desc="Use IDF:", options=[True, False,],),
            gui.dropdown(desc="Smooth IDF:", options=[True, False,],),
            gui.dropdown(desc="Sublinear TF:", options=[True, False,],),
            gui.dropdown(
                desc="View:", options=["Table", "Bar plot", "Horizontal bar plot",],
            ),
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

    def gui(self, **kwargs):

        self._panel[10]["widget"].disabled = kwargs["view"] == "Table"
        self._panel[11]["widget"].disabled = kwargs["view"] == "Table"

    def update(self, button):

        column = self._panel[0]["widget"].value
        top_n = self._panel[1]["widget"].value
        norm = self._panel[2]["widget"].value
        use_idf = self._panel[3]["widget"].value
        smooth_idf = self._panel[4]["widget"].value
        sublinear_tf = self._panel[5]["widget"].value
        view = self._panel[6]["widget"].value
        sort_by = self._panel[7]["widget"].value
        ascending = self._panel[8]["widget"].value
        cmap = self._panel[9]["widget"].value
        width = int(self._panel[10]["widget"].value)
        height = int(self._panel[11]["widget"].value)

        figsize = (width, height)

        recompute_TFIDF_matrix = False
        sort_TFIDF_matrix = False
        recompute_top_n = False

        if self._column != column:
            self._column = column
            matrix = TF_matrix(data=self._data, column=self._column)
            matrix = cmn.limit_to_exclude(
                data=matrix,
                axis=1,
                column=column,
                limit_to=self._limit_to,
                exclude=self._exclude,
            )
            self._TF_matrix = cmn.add_counters_to_axis(
                X=matrix, axis=1, data=self._data, column=column
            )
            recompute_TFIDF_matrix = True

        if self._norm != norm:
            self._norm = norm
            recompute_TFIDF_matrix = True

        if self._use_idf != use_idf:
            self._use_idf = use_idf
            recompute_TFIDF_matrix = True

        if self._smooth_idf != smooth_idf:
            self._smooth_idf = smooth_idf
            recompute_TFIDF_matrix = True

        if self._sublinear_tf != sublinear_tf:
            self._sublinear_tf = sublinear_tf
            recompute_TFIDF_matrix = True

        if recompute_TFIDF_matrix is True:
            norm = self._norm
            if norm is not None:
                norm = norm.lower()
            TFIDF_matrix_ = TFIDF_matrix(
                TF_matrix=self._TF_matrix,
                norm=norm,
                use_idf=self._use_idf,
                smooth_idf=self._smooth_idf,
                sublinear_tf=self._sublinear_tf,
            )
            vector = TFIDF_matrix_.sum(axis=0).sort_values(ascending=False).head(50)
            self._TFIDF_matrix_50 = vector.to_frame()
            self._TFIDF_matrix_50.columns = ["TF*IDF"]
            recompute_top_n = True

        if self._top_n != top_n or recompute_top_n is True:
            self._top_n = top_n
            self._TFIDF_matrix = self._TFIDF_matrix_50.head(top_n)
            sort_TFIDF_matrix = True

        if (
            self._sort_by != sort_by
            or self._ascending != ascending
            or sort_TFIDF_matrix is True
        ):
            self._sort_by = sort_by
            self._ascending = ascending
            if sort_by == "TF*IDF":
                self._TFIDF_matrix = self._TFIDF_matrix.sort_values(
                    "TF*IDF", ascending=ascending
                )
            else:
                self._TFIDF_matrix = cmn.sort_by_axis(
                    data=self._TFIDF_matrix,
                    sort_by=sort_by,
                    ascending=ascending,
                    axis=0,
                )

        figsize = (width, height)

        self._output.clear_output()
        with self._output:

            if view == "Table":

                display(self._TFIDF_matrix.style.background_gradient(cmap=cmap))

            if view == "Bar plot":

                display(
                    plt.bar(
                        height=self._TFIDF_matrix["TF*IDF"],
                        darkness=None,
                        cmap=cmap,
                        figsize=figsize,
                        fontsize=11,
                        ylabel="TF*IDF",
                    )
                )

            if view == "Horizontal bar plot":

                display(
                    plt.barh(
                        width=self._TFIDF_matrix["TF*IDF"],
                        darkness=None,
                        cmap=cmap,
                        figsize=figsize,
                        fontsize=11,
                        xlabel="TF*IDF",
                    )
                )
