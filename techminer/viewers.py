
"""
Data Viewer
==================================================================================================



"""
import ipywidgets as widgets
from ipywidgets import AppLayout, Layout
from techminer.explode import MULTIVALUED_COLS, __explode
import pandas as pd
from IPython.display import display

from techminer.by_term import summary_by_term

WIDGET_WIDTH = "200px"
LEFT_PANEL_HEIGHT = "588px"
RIGHT_PANEL_WIDTH = "870px"


COLUMNS = [
    "Author Keywords",
    "Authors",
    "Countries",
    "Country 1st",
    "Index Keywords",
    "Institution 1st",
    "Institutions",
    "Keywords",
    "Source title",
]

FIELDS = [
    "Authors",
    "Title",
    "Author Keywords",
    "Index Keywords",
    "Source title",
    "Year",
    "Countries",
    "Institutions",
    "Keywords",
    
]

def __body_0(df, top_n):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "column",
            "desc": "Column:",
            "widget": widgets.Dropdown(
                options=COLUMNS,
                value=COLUMNS[1],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "value",
            "desc": "Term:",
            "widget": widgets.Dropdown(
                options=[1], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "title",
            "desc": "Title:",
            "widget": widgets.Select(
                options=[1, 2, 3], layout=Layout(width="700px"),
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
        column = kwargs["column"]
        value = kwargs["value"]
        title = kwargs["title"]
        #
        # Populate value control with top_n terms
        #
        if column in MULTIVALUED_COLS:
            x = __explode(df[FIELDS], column)
        else:
            x = df[FIELDS]
        #
        if top_n is not None:
            summary = summary_by_term(df, column)
            top_terms_freq = set(summary.sort_values('Num Documents', ascending=False).head(top_n)[column])
            top_terms_cited_by = set(summary.sort_values('Cited by', ascending=False).head(top_n)[column])
            top_terms = sorted(top_terms_freq | top_terms_cited_by)
            controls[1]["widget"].options = top_terms
        else:
            all_terms = pd.Series(x[column].unique())
            all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
            all_terms = all_terms.sort_values()
            controls[1]["widget"].options = all_terms
        #
        # Populate titles
        #
        s = x[x[column] == controls[1]["widget"].value]
        controls[2]["widget"].options = sorted(s['Title'].tolist())
        #
        # Print info from selected title
        #
        out = df[df['Title'] == controls[2]["widget"].value]
        out = out.reset_index(drop=True)
        out = out.iloc[0]
        output.clear_output()
        text = ''
        with output:
            for f in FIELDS:
                z = out[f]
                if not pd.isna(z):
                    if f in MULTIVALUED_COLS:
                        v = z.split(';')
                        v = [a.strip() if isinstance(a, str) else a for a in v]
                        text += '{:>16}: {}<br>'.format(f, v[0])
                        for m in v[1:]:
                            text += '                  {}<br>'.format(m)
                    else:
                        text += '{:>16}: {}<br>'.format(f, z)
            display(widgets.HTML('<pre>' + text + '</pre>'))
            

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    widgets.interactive_output(server, args,  )
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Column Explorer (Top {} terms)".format(top_n)
            )),
        left_sidebar= widgets.VBox(
            [
                widgets.Label(value=controls[0]["desc"]),
                controls[0]["widget"],
                widgets.Label(value=controls[1]["desc"]),
                controls[1]["widget"]
            ],
            layout=Layout(width="200px")
        ),
        center=widgets.VBox([widgets.Label(value=controls[2]["desc"]), controls[2]["widget"]]),
        footer=widgets.VBox([output]),
        # pane_widths=[1,3,0],
        pane_heights=["80px", "130px", "550px"],
    )
    
    
def column(df, top_n=50):
    """Jupyter Lab dashboard.
    """
    return __body_0(df, top_n)

    
#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()