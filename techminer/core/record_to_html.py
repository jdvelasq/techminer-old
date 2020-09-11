import pandas as pd
import textwrap
import re


def record_to_HTML(x, only_abstract=False, keywords_to_highlight=None):
    """"""
    HTML = ""

    column_list = ["Title_HL" if "Title_HL" in x.index else "Title"]
    column_list += [
        "Year",
        "Authors",
        "Global_Citations",
    ]
    column_list += ["Abstract_HL" if "Abstract_HL" in x.index else "Abstract"]
    column_list += [
        "Author_Keywords",
        "Source_title",
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

            #
            # Highlight keywords
            #
            if keywords_to_highlight is not None and f in [
                "Author_Keywords",
                "Index_Keywords",
                "Author_Keywords_CL",
                "Index_Keywords_CL",
            ]:
                for keyword in keywords_to_highlight:
                    if keyword.lower() in z.lower():
                        pattern = re.compile(keyword, re.IGNORECASE)
                        z = pattern.sub("<b>" + keyword.upper() + "</b>", z)

            v = z.split(";")
            v = [a.strip() if isinstance(a, str) else a for a in v]
            HTML += "{:>18}: {}<br>".format(f, v[0])
            for m in v[1:]:
                HTML += " " * 20 + "{}<br>".format(m)
        else:
            if f == "Title" or f == "Abstract" or f == "Title_HL" or f == "Abstract_HL":

                #
                # Keywords to upper case
                #
                for keyword in keywords_to_highlight:
                    if keyword.lower() in z.lower():
                        pattern = re.compile(keyword, re.IGNORECASE)
                        z = pattern.sub("<b>" + keyword.upper() + "</b>", z)

                # if len(keywords_to_highlight) == 2 and f == "Title":
                #     keyword1 = keywords_to_highlight[0]
                #     keyword2 = keywords_to_highlight[1]
                #     if keyword1.lower() in z.lower() and keyword2.lower() in z.lower():
                #         z = '<b style="color:#FF6433">' + z + "</b>"

                if len(keywords_to_highlight) == 1 and f == "Abstract":
                    keyword = keywords_to_highlight[0]
                    z = z.split(". ")
                    z = [
                        '<b_style="color:#FF6433">' + w + "</b>"
                        if keyword.lower() in w.lower()
                        else w
                        for w in z
                    ]
                    z = ". ".join(z)

                if len(keywords_to_highlight) == 2 and f == "Abstract":
                    keyword1 = keywords_to_highlight[0]
                    keyword2 = keywords_to_highlight[1]
                    if not pd.isna(keyword1) and not pd.isna(keyword2):
                        z = z.split(". ")
                        z = [
                            '<b_style="color:#FF6433">' + w + "</b>"
                            if keyword1.lower() in w.lower()
                            and keyword2.lower() in w.lower()
                            else w
                            for w in z
                        ]
                        z = [
                            '<b_style="color:#2E86C1">' + w + "</b>"
                            if keyword1.lower() in w.lower()
                            and keyword2.lower() not in w.lower()
                            else w
                            for w in z
                        ]
                        z = [
                            '<b_style="color:#2EA405">' + w + "</b>"
                            if keyword1.lower() not in w.lower()
                            and keyword2.lower() in w.lower()
                            else w
                            for w in z
                        ]

                        z = ". ".join(z)

                s = textwrap.wrap(z, 80)
                HTML += "{:>18}: {}<br>".format(f, s[0])
                for t in s[1:]:
                    HTML += "{}<br>".format(textwrap.indent(t, " " * 20))
            elif f == "Global_Citations":
                HTML += "{:>18}: {}<br>".format(f, int(z))
            else:
                HTML += "{:>18}: {}<br>".format(f, z)

    HTML = HTML.replace("<b_style", "<b style")
    return "<pre>" + HTML + "</pre>"
