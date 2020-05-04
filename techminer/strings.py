"""
TechMiner.strings
===============================

This module contains functions for manipulating texts.



Functions in this module
----------------------------------------------------------------------------------------

"""


# import re
# import geopandas
# import string
# from nltk.stem import PorterStemmer

# # -------------------------------------------------------------------------------------------
# def find_string(pattern, x, ignore_case=True, full_match=False, use_re=False):
#     r"""Find pattern in string.

#     Args:
#         pattern (string)
#         x (string)
#         ignore_case (bool)
#         full_match (bool)
#         use_re (bool)

#     Returns:
#         string or None

#     >>> find_string(r'\btwo\b', 'one two three four five', use_re=True)
#     'two'

#     >>> find_string(r'\bTWO\b', 'one two three four five', use_re=True)
#     'two'

#     >>> find_string(r'\btwo\b', 'one TWO three four five', ignore_case=False, use_re=True) is None
#     True

#     >>> find_string(r'\btwo\Wthree\b', 'one two three four five', ignore_case=False, use_re=True)
#     'two three'

#     """

#     if use_re is False:
#         pattern = re.escape(pattern)

#     if full_match is True:
#         pattern = "^" + pattern + "$"

#     if ignore_case is True:
#         result = re.findall(pattern, x, re.I)
#     else:
#         result = re.findall(pattern, x)

#     if len(result):
#         return result[0]

#     return None


# # -------------------------------------------------------------------------------------------
# def replace_string(
#     pattern, x, repl=None, ignore_case=True, full_match=False, use_re=False
# ):
#     """Replace pattern in string.

#     Args:
#         pattern (string)
#         x (string)
#         repl (string, None)
#         ignore_case (bool)
#         full_match (bool)
#         use_re (bool)

#     Returns:
#         string or []

#     """

#     if use_re is False:
#         pattern = re.escape(pattern)

#     if full_match is True:
#         pattern = "^" + pattern + "$"

#     if ignore_case is True:
#         return re.sub(pattern, repl, x, re.I)
#     return re.sub(pattern, repl, x)


# # -------------------------------------------------------------------------------------------
# def extract_after_first(pattern, x, ignore_case=True, full_match=False, use_re=False):
#     """Returns the string from the first ocurrence of the keyword to the end of string x.

#     Args:
#         pattern (string) :
#         x : string

#     Returns:
#         String

#     >>> extract_after_first('aaa', '1 aaa 4 aaa 5')
#     'aaa 4 aaa 5'

#     >>> extract_after_first('bbb', '1 aaa 4 aaa 5')

#     """
#     y = find_string(pattern, x, ignore_case, full_match, use_re)

#     if y is not None:

#         if ignore_case is True:
#             c = re.compile(y, re.I)
#         else:
#             c = re.compile(y)

#         z = c.search(x)

#         if z:
#             return x[z.start() :]
#         else:
#             return None

#     return None


# # -------------------------------------------------------------------------------------------
# def extract_after_last(pattern, x, ignore_case=True, full_match=False, use_re=False):
#     """Returns the string from last ocurrence of a keyword to the end of string x.

#     Args:
#         x: string

#     Returns:
#         String

#     >>> extract_after_last('aaa', '1 aaa 4 aaa 5')
#     'aaa 5'

#     """

#     y = find_string(pattern, x, ignore_case, full_match, use_re)

#     if y is not None:

#         if ignore_case is True:
#             c = re.compile(y, re.I)
#         else:
#             c = re.compile(y)

#         z = c.findall(x)

#         result = x
#         for w in z[:-1]:
#             y = c.search(result)
#             result = result[y.end() :]
#         y = c.search(result)
#         return result[y.start() :]

#     return None


# # -------------------------------------------------------------------------------------------
# def extract_nearby(
#     pattern, x, n_words=1, ignore_case=True, full_match=False, use_re=False
# ):
#     """Extracts the words of string x in the proximity of the terms matching
#     the keywords list.

#     Args:
#         x (string): A string object.
#         n_words (integer): number of words around term.

#     Returns:
#         String.

#     **Examples**

#     >>> import pandas as pd
#     >>> df = pd.DataFrame({
#     ...    'f': ['1 2 3 4 5 6', 'aaa 1 2 3 4 5', '1 aaa 2 3 4 5', '1 2 aaa 3 4 5',
#     ...          '1 2 3 aaa 4 5', '1 2 3 4 aaa 5', '1 2 3 4 5 aaa'],
#     ... })
#     >>> df
#                    f
#     0    1 2 3 4 5 6
#     1  aaa 1 2 3 4 5
#     2  1 aaa 2 3 4 5
#     3  1 2 aaa 3 4 5
#     4  1 2 3 aaa 4 5
#     5  1 2 3 4 aaa 5
#     6  1 2 3 4 5 aaa
#     >>> df.f.map(lambda x: extract_nearby('aaa', x, n_words=2)) # doctest: +NORMALIZE_WHITESPACE
#     0           None
#     1        aaa 1 2
#     2      1 aaa 2 3
#     3    1 2 aaa 3 4
#     4    2 3 aaa 4 5
#     5      3 4 aaa 5
#     6        4 5 aaa
#     Name: f, dtype: object
#     """

#     def check(pattern):

#         if ignore_case is True:
#             c = re.compile(pattern, re.I)
#         else:
#             c = re.compile(pattern)

#         if full_match is True:
#             result = c.fullMatch(x)
#         else:
#             result = c.findall(x)

#         if len(result):
#             return result[0]
#         else:
#             return None

#     y = find_string(pattern, x, ignore_case, full_match, use_re)

#     if y is not None:

#         pattern = "\w\W" * n_words + y + "\W\w" * n_words
#         result = check(pattern)

#         if result is not None:
#             return result
#         else:
#             for i in range(n_words, -1, -1):

#                 # Checks at the beginning
#                 pattern = "^" + "\w\W" * i + y + "\W\w" * n_words
#                 result = check(pattern)
#                 if result is not None:
#                     return result

#             for j in range(n_words, -1, -1):
#                 # Checks at the end
#                 pattern = "\w\W" * n_words + y + "\W\w" * j + "$"
#                 result = check(pattern)
#                 if result is not None:
#                     return result

#             for i in range(n_words, -1, -1):
#                 for j in range(n_words, -1, -1):
#                     pattern = "^" + "\w\W" * i + y + "\W\w" * j + "$"
#                     result = check(pattern)
#                     if result is not None:
#                         return result[0]

#     return None


# # -------------------------------------------------------------------------------------------
# def extract_until_first(pattern, x, ignore_case=True, full_match=False, use_re=False):
#     """Returns the string from begining of x to the first ocurrence of a keyword.

#     Args:
#         x: string

#     Returns:
#         String

#     >>> extract_until_first('aaa', '1 aaa 4 aaa 5')
#     '1 aaa'

#     """
#     y = find_string(pattern, x, ignore_case, full_match, use_re)

#     if y is not None:

#         if ignore_case is True:
#             c = re.compile(y, re.I)
#         else:
#             c = re.compile(y)

#         z = c.search(x)

#         if z:
#             return x[: z.end()]
#         else:
#             return None

#     return None


# # -------------------------------------------------------------------------------------------
# def extract_until_last(pattern, x, ignore_case=True, full_match=False, use_re=False):
#     """Returns the string from begining of x to the last ocurrence of a keyword.

#     Args:
#         x: string

#     Returns:
#         String

#     >>> extract_until_last('aaa', '1 aaa 4 aaa 5')
#     '1 aaa 4 aaa'

#     """
#     y = find_string(pattern, x, ignore_case, full_match, use_re)

#     if y is not None:

#         if ignore_case is True:
#             c = re.compile("[\w+\W+]+" + y, re.I)
#         else:
#             c = re.compile("[\w+\W+]+" + y)

#         z = c.search(x)

#         if z:
#             return x[: z.end()]
#         else:
#             return None

#     return None


# # -------------------------------------------------------------------------------------------
# def extract_country(x, sep=";"):
#     """

#     >>> import pandas as pd
#     >>> x = pd.DataFrame({
#     ...     'Affiliations': [
#     ...         'University, Cuba; University, Venezuela',
#     ...         'University, United States; Univesity, Singapore',
#     ...         'University;',
#     ...         'University; Univesity',
#     ...         'University,',
#     ...         'University',
#     ...         None]
#     ... })
#     >>> x['Affiliations'].map(lambda x: extract_country(x))
#     0             Cuba;Venezuela
#     1    United States;Singapore
#     2                       None
#     3                       None
#     4                       None
#     5                       None
#     6                       None
#     Name: Affiliations, dtype: object
#     """

#     if x is None:
#         return None

#     #
#     # lista generica de nombres de paises
#     #
#     country_names = sorted(
#         geopandas.read_file(
#             geopandas.datasets.get_path("naturalearth_lowres")
#         ).name.tolist()
#     )

#     # paises faltantes
#     country_names.append("Singapore")
#     country_names.append("Malta")
#     country_names.append("United States")

#     #
#     #  Reemplazo de nombres de regiones administrativas
#     # por nombres de paises
#     #
#     x = re.sub("Bosnia and Herzegovina", "Bosnia and Herz.", x)
#     x = re.sub("Czech Republic", "Czechia", x)
#     x = re.sub("Russian Federation", "Russia", x)
#     x = re.sub("Hong Kong", "China", x)
#     x = re.sub("Macau", "China", x)
#     x = re.sub("Macao", "China", x)

#     countries = [affiliation.split(",")[-1].strip() for affiliation in x.split(sep)]

#     countries = ";".join(
#         [country if country in country_names else "" for country in countries]
#     )

#     if countries == "" or countries == ";":
#         return None
#     else:
#         return countries


# # -------------------------------------------------------------------------------------------
# def steamming(pattern, text):

#     text = asciify(text)
#     pattern = asciify(pattern)

#     text = text.strip().lower()
#     pattern = pattern.strip().lower()

#     porter = PorterStemmer()

#     pattern = [porter.stem(w) for w in pattern.split()]
#     text = [porter.stem(w) for w in text.split()]

#     return [m in text for m in pattern]


# # -------------------------------------------------------------------------------------------
# def steamming_all(pattern, text):
#     """

#     >>> steamming_all('computers cars', 'car computing')
#     True

#     >>> steamming_all('computers cars', 'car houses')
#     False

#     """
#     return all(steamming(pattern, text))


# # -------------------------------------------------------------------------------------------
# def steamming_any(pattern, text):
#     """

#     >>> steamming_any('computers cars', 'car computing')
#     True

#     >>> steamming_any('computers cars', 'computing house')
#     True

#     >>> steamming_all('computers cars', 'tree houses')
#     False

#     """
#     return any(steamming(pattern, text))


# # -------------------------------------------------------------------------------------------
# def fingerprint(x):
#     """Computes 'fingerprint' representation of string x.

#     Args:
#         x (string): string to convert.

#     Returns:
#         string.

#     **Examples**

#     >>> fingerprint('a A b')
#     'a b'
#     >>> fingerprint('b a a')
#     'a b'
#     >>> fingerprint(None) is None
#     True
#     >>> fingerprint('b c')
#     'b c'
#     >>> fingerprint(' c b ')
#     'b c'


#     """
#     porter = PorterStemmer()

#     if x is None:
#         return None
#     x = x.strip().lower()
#     x = re.sub("-", " ", x)
#     x = re.sub("[" + string.punctuation + "]", "", x)
#     x = asciify(x)
#     x = " ".join(porter.stem(w) for w in x.split())
#     x = " ".join({w for w in x.split()})
#     x = " ".join(sorted(x.split(" ")))
#     return x


# # -------------------------------------------------------------------------------------------


def remove_accents(text):
    """Translate non-ascii charaters to ascii equivalent. Based on Google Open Refine.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> remove_accents('áéíóúñÁÉÍÓÚÑ')
    'aeiounAEIOUN'

    """

    def translate(c):

        if c in [
            "\u0100",
            "\u0102",
            "\u00C5",
            "\u0104",
            "\u00C0",
            "\u00C1",
            "\u00C2",
            "\u00C3",
            "\u00C4",
        ]:
            return "A"

        if c in [
            "\u00E0",
            "\u00E1",
            "\u00E2",
            "\u00E3",
            "\u00E4",
            "\u0103",
            "\u0105",
            "\u00E5",
            "\u0101",
        ]:
            return "a"

        if c in [
            "\u00C7",
            "\u0106",
            "\u0108",
            "\u010A",
            "\u010C",
        ]:
            return "C"

        if c in [
            "\u010D",
            "\u00E7",
            "\u0107",
            "\u010B",
            "\u0109",
        ]:
            return "c"

        if c in [
            "\u00D0",
            "\u010E",
            "\u0110",
        ]:
            return "D"

        if c in [
            "\u0111",
            "\u00F0",
            "\u010F",
        ]:
            return "d"

        if c in [
            "\u00C8",
            "\u00C9",
            "\u00CA",
            "\u00CB",
            "\u0112",
            "\u0114",
            "\u0116",
            "\u0118",
            "\u011A",
        ]:
            return "E"

        if c in [
            "\u011B",
            "\u0119",
            "\u00E8",
            "\u00E9",
            "\u00EA",
            "\u00EB",
            "\u0113",
            "\u0115",
            "\u0117",
        ]:
            return "e"

        if c in [
            "\u011C",
            "\u011E",
            "\u0120",
            "\u0122",
        ]:
            return "G"

        if c in [
            "\u0123",
            "\u011D",
            "\u011F",
            "\u0121",
        ]:
            return "g"

        if c in [
            "\u0124",
            "\u0126",
        ]:
            return "H"

        if c in [
            "\u0127",
            "\u0125",
        ]:
            return "h"

        if c in [
            "\u00CC",
            "\u00CD",
            "\u00CE",
            "\u00CF",
            "\u0128",
            "\u012A",
            "\u012C",
            "\u012E",
            "\u0130",
        ]:
            return "I"

        if c in [
            "\u0131",
            "\u012F",
            "\u012D",
            "\u00EC",
            "\u012B",
            "\u0129",
            "\u00EF",
            "\u00EE",
            "\u00ED",
            "\u017F",
        ]:
            return "i"

        if c in [
            "\u0134",
        ]:
            return "J"
        if c in [
            "\u0135",
        ]:
            return "j"

        if c in [
            "\u0136",
        ]:
            return "K"

        if c in [
            "\u0137",
            "\u0138",
        ]:
            return "k"

        if c in [
            "\u0139",
            "\u013B",
            "\u013D",
            "\u013F",
            "\u0141",
        ]:
            return "L"

        if c in [
            "\u0142",
            "\u013A",
            "\u013C",
            "\u013E",
            "\u0140",
        ]:
            return "l"

        if c in [
            "\u00D1",
            "\u0143",
            "\u0145",
            "\u0147",
        ]:
            return "N"

        if c in [
            "\u014B",
            "\u014A",
            "\u0149",
            "\u0148",
            "\u0146",
            "\u0144",
            "\u00F1",
        ]:
            return "n"

        if c in [
            "\u00D2",
            "\u00D3",
            "\u00D4",
            "\u00D5",
            "\u00D6",
            "\u00D8",
            "\u014C",
            "\u014E",
            "\u0150",
        ]:
            return "O"

        if c in [
            "\u0151",
            "\u00F2",
            "\u00F3",
            "\u00F4",
            "\u00F5",
            "\u00F6",
            "\u00F8",
            "\u014F",
            "\u014D",
        ]:
            return "o"

        if c in [
            "\u0154",
            "\u0156",
            "\u0158",
        ]:
            return "R"

        if c in [
            "\u0159",
            "\u0155",
            "\u0157",
        ]:
            return "r"

        if c in [
            "\u015A",
            "\u015C",
            "\u015E",
            "\u0160",
        ]:
            return "S"

        if c in [
            "\u0161",
            "\u015B",
            "\u015F",
            "\u015D",
        ]:
            return "s"

        if c in [
            "\u0162",
            "\u0164",
            "\u0166",
        ]:
            return "T"

        if c in [
            "\u0167",
            "\u0163",
            "\u0165",
        ]:
            return "t"

        if c in [
            "\u00D9",
            "\u00DA",
            "\u00DB",
            "\u00DC",
            "\u0168",
            "\u016A",
            "\u016E",
            "\u0170",
            "\u0172",
            "\u016C",
        ]:
            return "U"

        if c in [
            "\u0173",
            "\u00F9",
            "\u00FA",
            "\u00FB",
            "\u00FC",
            "\u0169",
            "\u016B",
            "\u016D",
            "\u0171",
            "\u016F",
        ]:
            return "u"

        if c in [
            "\u0174",
        ]:
            return "W"

        if c in [
            "\u0175",
        ]:
            return "w"

        if c in [
            "\u0178",
            "\u00DD",
            "\u0176",
        ]:
            return "Y"

        if c in [
            "\u0177",
            "\u00FD",
            "\u00FF",
        ]:
            return "y"

        if c in [
            "\u0179",
            "\u017B",
            "\u017D",
        ]:
            return "Z"

        if c in [
            "\u017E",
            "\u017A",
            "\u017C",
        ]:
            return "z"

        return c

    return "".join([translate(c) for c in text])
