"""
Analysis by Year
==================================================================================================



"""
import ipywidgets as widgets
import numpy as np
import pandas as pd
import techminer.plots as plt
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, Layout
from techminer.plots import COLORMAPS


def summary_by_year(df):
    """Computes the number of document and the number of total citations per year.
    This funciton adds the missing years in the sequence.


    Args:
        df (pandas.DataFrame): bibliographic dataframe.


    Returns:
        pandas.DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "Cited_by": list(range(10,16)),
    ...          "ID": list(range(6)),
    ...     }
    ... )
    >>> df
       Year  Cited_by  ID
    0  2010        10   0
    1  2010        11   1
    2  2011        12   2
    3  2011        13   3
    4  2012        14   4
    5  2016        15   5

    >>> summary_by_year(df)[['Year', 'Cited_by', 'Num_Documents', 'ID']]
       Year  Cited by  Num Documents      ID
    0  2010        21              2  [0, 1]
    1  2011        25              2  [2, 3]
    2  2012        14              1     [4]
    3  2013         0              0      []
    4  2014         0              0      []
    5  2015         0              0      []
    6  2016        15              1     [5]

    >>> summary_by_year(df)[['Cum_Num_Documents', 'Cum_Cited_by', 'Avg_Cited_by']]
         CUM_Num_Documents    Cum_Cited_by   Avg_Cited_by
    0                    2              21           10.5
    1                    4              46           12.5
    2                    5              60           14.0
    3                    5              60            0.0
    4                    5              60            0.0
    5                    5              60            0.0
    6                    6              75           15.0

    """
    data = df[["Year", "Cited_by", "ID"]].explode("Year")
    data["Num_Documents"] = 1
    result = data.groupby("Year", as_index=False).agg(
        {"Cited_by": np.sum, "Num_Documents": np.size}
    )
    result = result.assign(
        ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
    )
    result["Cited_by"] = result["Cited_by"].map(lambda x: int(x))
    years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
    result = result.set_index("Year")
    result = result.reindex(years, fill_value=0)
    result["ID"] = result["ID"].map(lambda x: [] if x == 0 else x)
    result.sort_values(
        "Year", ascending=True, inplace=True,
    )
    result["Cum_Num_Documents"] = result["Num_Documents"].cumsum()
    result["Cum_Cited_by"] = result["Cited_by"].cumsum()
    result["Avg_Cited_by"] = result["Cited_by"] / result["Num_Documents"]
    result["Avg_Cited_by"] = result["Avg_Cited_by"].map(
        lambda x: 0 if pd.isna(x) else x
    )
    result = result.reset_index()
    return result


def documents_by_year(x, cumulative=False):
    """Computes the number of documents per year.
    This function adds the missing years in the sequence.

    Args:
        cumulative (bool): cumulate values per year.

    Returns:
        DataFrame.

    """
    result = summary_by_year(x, cumulative)
    result.pop("Cited_by")
    result = result.reset_index(drop=True)
    return result


def citations_by_year(x, cumulative=False):
    """Computes the number of citations by year.
    This function adds the missing years in the sequence.

    Args:
        cumulative (bool): cumulate values per year.

    Returns:
        DataFrame.

    """
    result = summary_by_year(x, cumulative)
    result.pop("Num_Documents")
    result = result.reset_index(drop=True)
    return result

    def growth_indicators(self, column, sep=None, timewindow=2, keywords=None):
        """Computes the average growth rate of a group of terms.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            timewindow (int): time window for analysis
            keywords (Keywords): filter the result using the specified Keywords object.

        Returns:
            DataFrame.


        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2013, 2014, 2014],
        ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4,author 0;author 3,author 3;author 4".split(","),
        ...          "Cited_by": list(range(10,18)),
        ...          "ID": list(range(8)),
        ...     }
        ... )
        >>> df
           Year                     Authors  Cited by  ID
        0  2010  author 0;author 1;author 2        10   0
        1  2010                    author 0        11   1
        2  2011                    author 1        12   2
        3  2011                    author 3        13   3
        4  2012                    author 4        14   4
        5  2013                    author 4        15   5
        6  2014           author 0;author 3        16   6
        7  2014           author 3;author 4        17   7

        >>> DataFrame(df).documents_by_term_per_year('Authors', as_matrix=True)
              author 0  author 1  author 2  author 3  author 4
        2010         2         1         1         0         0
        2011         0         1         0         1         0
        2012         0         0         0         0         1
        2013         0         0         0         0         1
        2014         1         0         0         2         1

        >>> DataFrame(df).growth_indicators('Authors')
            Authors       AGR  ADY   PDLY  Before 2013  Between 2013-2014
        0  author 3  0.666667  1.0  12.50            1                  2
        1  author 0  0.333333  0.5   6.25            2                  1
        2  author 4  0.000000  1.0  12.50            1                  2

        >>> keywords = Keywords(['author 3', 'author 4'])
        >>> keywords = keywords.compile()
        >>> DataFrame(df).growth_indicators('Authors', keywords=keywords)
            Authors       AGR  ADY  PDLY  Before 2013  Between 2013-2014
        0  author 3  0.666667  1.0  12.5            1                  2
        1  author 4  0.000000  1.0  12.5            1                  2

        """

        def compute_agr():
            result = self.documents_by_term_per_year(
                column=column, sep=sep, keywords=keywords
            )
            years_agr = sorted(set(result.Year))[-(timewindow + 1) :]
            years_agr = [years_agr[0], years_agr[-1]]
            result = result[result.Year.map(lambda w: w in years_agr)]
            result.pop("ID")
            result = pd.pivot_table(
                result,
                columns="Year",
                index=column,
                values="Num_Documents",
                fill_value=0,
            )
            result["AGR"] = 0.0
            result = result.assign(
                AGR=(result[years_agr[1]] - result[years_agr[0]]) / (timewindow + 1)
            )
            result.pop(years_agr[0])
            result.pop(years_agr[1])
            result = result.reset_index()
            result.columns = list(result.columns)
            result = result.sort_values(by=["AGR", column], ascending=False)
            return result

        def compute_ady():
            result = self.documents_by_term_per_year(
                column=column, sep=sep, keywords=keywords
            )
            years_ady = sorted(set(result.Year))[-timewindow:]
            result = result[result.Year.map(lambda w: w in years_ady)]
            result = result.groupby([column], as_index=False).agg(
                {"Num_Documents": np.sum}
            )
            result = result.set_index(column)
            result = result.rename(columns={"Num_Documents": "ADY"})
            result["ADY"] = result.ADY.map(lambda w: w / timewindow)
            return result.ADY

        def compute_num_documents():
            result = self.documents_by_term_per_year(
                column=column, sep=sep, keywords=keywords
            )
            years_between = sorted(set(result.Year))[-timewindow:]
            years_before = sorted(set(result.Year))[0:-timewindow]
            between = result[result.Year.map(lambda w: w in years_between)]
            before = result[result.Year.map(lambda w: w in years_before)]
            between = between.groupby([column], as_index=False).agg(
                {"Num_Documents": np.sum}
            )
            between = between.rename(
                columns={
                    "Num_Documents": "Between {}-{}".format(
                        years_between[0], years_between[-1]
                    )
                }
            )
            before = before.groupby([column], as_index=False).agg(
                {"Num_Documents": np.sum}
            )
            before = before.rename(
                columns={"Num_Documents": "Before {}".format(years_between[0])}
            )
            result = pd.merge(before, between, on=column)
            result = result.set_index(column)
            return result

        result = compute_agr()
        result = result.set_index(column)
        ady = compute_ady()
        result.at[ady.index, "ADY"] = ady
        result = result.assign(PDLY=round(result.ADY / len(self) * 100, 2))
        num_docs = compute_num_documents()
        result = pd.merge(result, num_docs, on=column)
        result = result.reset_index()
        return result


#
#
#  APP
#
#

WIDGET_WIDTH = "200px"
LEFT_PANEL_HEIGHT = "588px"
RIGHT_PANEL_WIDTH = "870px"
FIGSIZE = (15, 9.4)
PANE_HEIGHTS = ["80px", "650px", 0]


def __body_0(df):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "selected_plot",
            "desc": "Plot:",
            "widget": widgets.Dropdown(
                options=[
                    "Documents by Year",
                    "Cum. Documents by Year",
                    "Times Cited by Year",
                    "Cum. Times Cited by Year",
                    "Avg. Times Cited by Year",
                ],
                value="Documents by Year",
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "plot_type",
            "desc": "Plot type:",
            "widget": widgets.Dropdown(
                options=["bar", "barh"], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "cmap",
            "desc": "Colormap:",
            "widget": widgets.Dropdown(
                options=COLORMAPS, layout=Layout(width=WIDGET_WIDTH),
            ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        # Logic
        #
        plots = {"bar": plt.bar, "barh": plt.barh}
        data = {
            "Documents by Year": ["Year", "Num_Documents"],
            "Cum. Documents by Year": ["Year", "Cum_Num_Documents"],
            "Times Cited by Year": ["Year", "Cited_by"],
            "Cum. Times Cited by Year": ["Year", "Cum_Cited_by"],
            "Avg. Times Cited by Year": ["Year", "Avg_Cited_by"],
        }
        #
        x = summary_by_year(df)
        x = x[data[kwargs["selected_plot"]]]
        plot = plots[kwargs["plot_type"]]
        #
        output.clear_output()
        with output:
            display(plot(x, cmap=kwargs["cmap"], figsize=FIGSIZE))

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    widgets.interactive_output(
        server, args,
    )
    return widgets.HBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [widgets.Label(value=control["desc"]), control["widget"]]
                    )
                    for control in controls
                ],
                layout=Layout(height=LEFT_PANEL_HEIGHT, border="1px solid gray"),
            ),
            widgets.VBox(
                [output], layout=Layout(width=RIGHT_PANEL_WIDTH, align_items="baseline")
            ),
        ]
    )


def app(df):
    """Jupyter Lab dashboard.
    """
    #
    body = widgets.Tab()
    body.children = [__body_0(df)]
    body.set_title(0, "Time Analysis")
    #
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Summary by Year"
            )
        ),
        center=body,
        pane_heights=PANE_HEIGHTS,
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
