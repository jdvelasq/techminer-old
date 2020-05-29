"""
Jupyter Lab Interface
==================================================================================================


"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

import techminer.analytics as tc
import techminer.plots as plt


def summary_by_year(x, **kwargs):
    """Computes the number of document and the number of total citations per year.
    This funciton adds the missing years in the sequence.

    
    Args:
        x (pandas.DataFrame): bibliographic dataframe.
        

    Returns:
        pandas.DataFrame.
    """
    df = tc.summary_by_year(x)

    #
    # Documents by year tab
    #
    docs_by_year = widgets.Output()
    with docs_by_year:
        display(plt.bar(df[["Year", "Num Documents"]], **kwargs))
    #
    # Citations by year tab
    #
    citations_by_year = widgets.Output()
    with citations_by_year:
        display(plt.bar(df[["Year", "Cited by"]], **kwargs))

    #
    # Jupyter Lab interface
    #
    widget = widgets.Tab()
    widget.children = [docs_by_year, citations_by_year]
    widget.set_title(0, "Documents by year")
    widget.set_title(1, "Citations by year")
    return widget
