import ipywidgets as widgets
from IPython.display import HTML, clear_output, display
from ipywidgets import AppLayout, GridspecLayout, Layout

import techminer.by_term as by_term
import pandas as pd

#
#
#  Top documents
#
#
def top_documents(data):
    """Returns the top 50 documents by Times Cited.

    Args:
        data (pandas.DataFrame): A bibliographic dataframe.

    Returns:
    """
    data = data.sort_values(["Times_Cited", "Year"], ascending=[False, True])
    data = data.head(50)
    data["Times_Cited"] = data.Times_Cited.map(lambda w: int(w))
    data = data.reset_index(drop=True)
    data = data.sort_values(["Times_Cited", "Title"], ascending=[False, True])
    data = data[["Authors", "Year", "Title", "Source_title", "Times_Cited"]]
    data["Times_Cited"] = data.Times_Cited.map(lambda w: int(w))
    data = data.reset_index(drop=True)

    return data


# def __APP4__(data):
#     """
#     # >>> import pandas as pd
#     # >>> data = pd.DataFrame(
#     # ...     {
#     # ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
#     # ...          "Times_Cited": list(range(10,16)),
#     # ...          "ID": list(range(6)),
#     # ...     }
#     # ... )
#     # >>> __APP4__(data)


#     """
#     output = widgets.Output()
#     with output:
#         display(result)
#     grid = GridspecLayout(10, 8)
#     grid[0:, 0:] = widgets.VBox(
#         [output], layout=Layout(height="657px", border="2px solid gray")
#     )
#     return grid

