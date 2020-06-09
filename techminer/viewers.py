
"""
Data Viewer
==================================================================================================


"""
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipywidgets import AppLayout, GridspecLayout, Layout
import textwrap

from techminer.by_term import summary_by_term
from techminer.explode import __explode

WIDGET_WIDTH = "200px"

def __record_to_HTML(x):
    """
    """
    HTML = ''
    for f in [
        "Title",
        "Authors",
        "Author_Keywords",
        "Index_Keywords",
        "Author_Keywords_CL",
        "Index_Keywords_CL",
        "Source_title",
        "Year",
        "Abstract",
        "Countries",
        "Institutions",
        "Cited_by"
        
    ]:
        z = x[f]
        if pd.isna(z) is True:
            continue
        if f in [
            "Authors",
            "Author_Keywords",
            "Index_Keywords",
            "Author_Keywords_CL",
            "Index_Keywords_CL",
            "Countries",
            "Institutions",
            "Source_title",
        ]:
            v = z.split(';')
            v = [a.strip() if isinstance(a, str) else a for a in v]
            HTML += '{:>18}: {}<br>'.format(f, v[0])
            for m in v[1:]:
                HTML += ' ' * 20 + '{}<br>'.format(m)
        else:
            if f == "Title" or f == "Abstract":
                s = textwrap.wrap(z, 80)
                HTML += '{:>18}: {}<br>'.format(f, s[0])
                for t in s[1:]:
                    HTML += '{}<br>'.format(textwrap.indent(t, ' ' * 20))
            elif f == 'Cited_by':
                HTML += '{:>18}: {}<br>'.format(f, int(z))
            else:
                HTML += '{:>18}: {}<br>'.format(f, z)
    return '<pre>' + HTML + '</pre>'

def column(df, top_n=50):
    """Jupyter Lab dashboard.
    """
    COLUMNS = [
        "Author_Keywords",
        "Author_Keywords_CL",
        "Authors",
        "Countries",
        "Country_1st_Author",
        "Index_Keywords",
        "Index_Keywords_CL",
        "Institution_1st_Author",
        "Institutions",
        "Keywords",
        "Source_title",
        "Title",
        "Year",
    ]
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
                options=[z for z in COLUMNS if z in df.columns],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "value",
            "desc": "Term:",
            "widget": widgets.Dropdown(
                options=[], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "title",
            "desc": "Title:",
            "widget": widgets.Select(
                options=[], layout=Layout(width="700px"),
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
        x = __explode(df, column)
        if top_n is not None:
            summary = summary_by_term(df, column)
            top_terms_freq = set(summary.sort_values('Num_Documents', ascending=False).head(top_n)[column])
            top_terms_cited_by = set(summary.sort_values('Cited_by', ascending=False).head(top_n)[column])
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
        with output:
            display(widgets.HTML(__record_to_HTML(out)))

            

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
        pane_heights=["80px", "130px", "550px"],
    )
    
    

def matrix(df, top_n=50):
    """Jupyter Lab dashboard to matrix data.
    """
    COLUMNS = [c for c in df.columns if c[0] != '*' and c not in ['Title', 'ID', 'Cited by', 'Abstract']]
    #
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    controls = [
        # 0
        {
            "arg": "main_column",
            "desc": "Column:",
            "widget": widgets.Dropdown(
                options=COLUMNS,
                value=COLUMNS[0],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 1
        {
            "arg": "main_term",
            "desc": "Term in Column:",
            "widget": widgets.Dropdown(
                options=[], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 2
        {
            "arg": "by_column",
            "desc": "By column:",
            "widget": widgets.Dropdown(
                options=COLUMNS,
                value=COLUMNS[0],
                layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 3
        {
            "arg": "by_term",
            "desc": "Term in By column:",
            "widget": widgets.Dropdown(
                options=[], layout=Layout(width=WIDGET_WIDTH),
            ),
        },
        # 4
        {
            "arg": "title",
            "desc": "Title:",
            "widget": widgets.Select(
                options=[], layout=Layout(width="500px"),
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
        main_column = kwargs["main_column"]
        main_term = kwargs["main_term"]
        by_column = kwargs["by_column"]
        by_term = kwargs["by_term"]
        title = kwargs["title"]
        #
        # Populate main_column with top_n terms
        #
        xdf = df.copy()
        #xdf = xdf[FIELDS]
        xdf['_key1_'] = xdf[main_column]
        xdf['_key2_'] = xdf[by_column]
        if main_column in MULTIVALUED_COLS:
            xdf = __explode(xdf, '_key1_')
        #
        if top_n is not None:
            summary = summary_by_term(xdf, '_key1_')
            top_terms_freq = set(summary.sort_values('Num Documents', ascending=False).head(top_n)['_key1_'])
            top_terms_cited_by = set(summary.sort_values('Cited by', ascending=False).head(top_n)['_key1_'])
            top_terms = sorted(top_terms_freq | top_terms_cited_by)
            controls[1]["widget"].options = top_terms
        else:
            top_terms = pd.Series(xdf['_key1_'].unique())
            top_terms = top_terms[top_terms.map(lambda w: not pd.isna(w))]
            top_terms = top_terms.sort_values()
            controls[1]["widget"].options = top_terms
        #
        # Subset selection
        #
        if by_column in MULTIVALUED_COLS:
            xdf = __explode(xdf, '_key2_')
        xdf = xdf[xdf['_key1_'] == controls[1]["widget"].value]
        terms = sorted(pd.Series(xdf['_key2_'].dropna().unique()))
        controls[3]["widget"].options = terms
        #
        # Title
        #
        xdf = xdf[xdf['_key2_'] == controls[3]["widget"].value]
        if len(xdf): 
            controls[4]["widget"].options = sorted(xdf['Title'].tolist())
        else:
            controls[4]["widget"].options = []
        #
        # Print info from selected title
        #
        out = df[df['Title'] == controls[4]["widget"].value]
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
                        text += '{:>25}: {}<br>'.format(f, v[0])
                        for m in v[1:]:
                            text += '                           {}<br>'.format(m)
                    else:
                        text += '{:>25}: {}<br>'.format(f, z)
            display(widgets.HTML('<pre>' + text + '</pre>'))
   

    # -------------------------------------------------------------------------
    #
    # Generic
    #
    # -------------------------------------------------------------------------
    args = {control["arg"]: control["widget"] for control in controls}
    output = widgets.Output()
    widgets.interactive_output(server, args,)
    
    #
    #
    grid = GridspecLayout(10, 5, height='800px')
    grid[0,:] = widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                "Matrix Explorer (Top {} terms)".format(top_n)
            ))
    grid[1,0] =  widgets.VBox(
            [
                widgets.Label(value=controls[0]["desc"]),
                controls[0]["widget"],
            ])

    grid[2,0] = widgets.VBox(
            [
                widgets.Label(value=controls[1]["desc"]),
                controls[1]["widget"]
            ])            

    grid[1,1] = widgets.VBox(
            [
                widgets.Label(value=controls[2]["desc"]),
                controls[2]["widget"],
            ])

    grid[2,1] = widgets.VBox(
            [
                widgets.Label(value=controls[3]["desc"]),
                controls[3]["widget"]
            ])
    
    grid[1:3, 2:] = widgets.VBox([widgets.Label(value=controls[4]["desc"]), controls[4]["widget"]])

    grid[3:, :] = widgets.VBox([output])
    
    return grid
    



    # return AppLayout(
    #     header=widgets.HTML(
    #         value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
    #             "Matrix Explorer (Top {} terms)".format(top_n)
    #         )),
    #     left_sidebar=
    #         widgets.VBox(
    #         [
    #             widgets.Label(value=controls[0]["desc"]),
    #             controls[0]["widget"],
    #             widgets.Label(value=controls[1]["desc"]),
    #             controls[1]["widget"]
    #         ],
    #         layout=Layout(width="200px")
    #     ),
    #     center = widgets.VBox(
    #         [
    #             widgets.Label(value=controls[2]["desc"]),
    #             controls[2]["widget"],
    #             widgets.Label(value=controls[3]["desc"]),
    #             controls[3]["widget"]
    #         ],
    #         layout=Layout(width="200px")
    #     ),
    #     right_sidebar=widgets.VBox([widgets.Label(value=controls[4]["desc"]), controls[4]["widget"]]),
    #     footer=widgets.VBox([output]),
    #     pane_heights=["80px", "130px", "550px"],
    # )
    
    
#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
