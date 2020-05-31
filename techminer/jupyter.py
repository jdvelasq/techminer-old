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


def summary_by_term(x, top_n=10, **kwargs):
    """
    """
    columns = [
        ("AU", "Authors"),
        ("SO", "Source titles"),
        ("ID", "Index Keywords "),
        ("DE", "Author Keywords"),
        ("DE_ID", "Keywords"),
        ("AU_CO", "Country"),
        ("AU_IN", "Institution"),
    ]
    #
    # Documents by term
    #
    widget_tab0_children = []
    for i in range(len(columns)):
        y = tc.documents_by_term(x, columns[i][0])
        z = y[[y.columns[0], "Num Documents"]].head(top_n)
        w = widgets.Output()
        with w:
            display(plt.bar(z, **kwargs))
        widget_tab0_children.append(w)
        #
    widget_tab0 = widgets.Tab()
    widget_tab0.children = widget_tab0_children
    for i in range(len(columns)):
        widget_tab0.set_title(i, columns[i][1])
    #
    # Citations by term
    #
    widget_tab1_children = []
    for i in range(len(columns)):
        y = tc.citations_by_term(x, columns[i][0])
        z = y[[y.columns[0], "Times Cited"]].head(top_n)
        w = widgets.Output()
        with w:
            display(plt.bar(z, **kwargs))
        widget_tab1_children.append(w)
        #
    widget_tab1 = widgets.Tab()
    widget_tab1.children = widget_tab1_children
    for i in range(len(columns)):
        widget_tab1.set_title(i, columns[i][1])
    #
    # Worldmaps
    #
    widget_tab2_children = []
    y = tc.documents_by_term(x, "AU_CO")
    y = y[["Country", "Num Documents"]]
    w = widgets.Output()
    with w:
        display(plt.worldmap(y, **kwargs))
    widget_tab2_children.append(w)
    #
    y = tc.citations_by_term(x, "AU_CO")
    y = y[["Country", "Times Cited"]]
    w = widgets.Output()
    with w:
        display(plt.worldmap(y, **kwargs))
    widget_tab2_children.append(w)
    widget_tab2 = widgets.Tab()
    widget_tab2.children = widget_tab2_children
    widget_tab2.set_title(0, "Documents by country")
    widget_tab2.set_title(1, "Citations by country")
    #
    # Accordion Widget
    #
    accordion = widgets.Accordion()
    accordion.children = [widget_tab0, widget_tab1, widget_tab2]
    accordion.set_title(0, "Documents by term")
    accordion.set_title(1, "Citations by term")
    accordion.set_title(2, "Worldmaps")
    return accordion
