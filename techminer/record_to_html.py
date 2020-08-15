import pandas as pd
import textwrap


def record_to_HTML(x, only_abstract=False):
    """
    """
    HTML = ""

    column_list = [
        "Title",
        "Authors",
        "Abstract",
        "Author_Keywords",
        "Index_Keywords",
        "Source_title",
        "Year",
        "Times_Cited",
    ]

    if only_abstract is False:
        column_list += [
            "Author_Keywords_CL",
            "Index_Keywords_CL",
            "Title_words",
            "Title_words_CL",
            "Abstract_words",
            "Abstract_words_CL",
            "Countries",
            "Institutions",
        ]

    for f in column_list:
        if f not in x.index:
            continue
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
            "Abstract_words",
            "Abstract_words_CL",
            "Title_words",
            "Title_words_CL",
        ]:
            v = z.split(";")
            v = [a.strip() if isinstance(a, str) else a for a in v]
            HTML += "{:>18}: {}<br>".format(f, v[0])
            for m in v[1:]:
                HTML += " " * 20 + "{}<br>".format(m)
        else:
            if f == "Title" or f == "Abstract":
                s = textwrap.wrap(z, 80)
                HTML += "{:>18}: {}<br>".format(f, s[0])
                for t in s[1:]:
                    HTML += "{}<br>".format(textwrap.indent(t, " " * 20))
            elif f == "Times_Cited":
                HTML += "{:>18}: {}<br>".format(f, int(z))
            else:
                HTML += "{:>18}: {}<br>".format(f, z)
    return "<pre>" + HTML + "</pre>"

