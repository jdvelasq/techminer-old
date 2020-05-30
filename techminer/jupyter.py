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
    data = [
        (["Year", "Num Documents"], "Documents by Year"),
        (["Year", "Num Documents (Cum)"], "Cum. Documents by Year"),
        (["Year", "Times Cited"], "Times Cited by Year"),
        (["Year", "Times Cited (Cum)"], "Cum. Times Cited by Year"),
        (["Year", "Avg. Times Cited"], "Avg. Times Cited by Year"),
    ]
    widget_list = []
    for i in range(len(data)):
        w = widgets.Output()
        with w:
            display(plt.bar(df[data[i][0]], **kwargs))
        widget_list.append(w)
    widget = widgets.Tab()
    widget.children = widget_list
    for i in range(len(data)):
        widget.set_title(i, data[i][1])
    return widget
