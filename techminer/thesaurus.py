"""
Thesaurus
==================================================================================================

"""
# import json
# import re

# import pandas as pd

# from techminer.text import (
#     fingerprint,
#     one_gram,
#     two_gram,
#     stemmer_porter,
#     stemmer_snowball,
# )


# def text_clustering(
#     x, sep=None, name_strategy="mostfrequent", key="porter", transformer=None
# ):
#     """Builds a thesaurus by clustering a list of strings.

#     Args:
#         x (list): list  of string to create thesaurus.

#         name_strategy (string): method for assigning keys in thesaurus.

#             * 'mostfrequent': Most frequent string in the cluster.

#             * 'longest': Longest string in the cluster.

#             * 'shortest': Shortest string in the cluster.

#         key (str): 'fingerprint', '1-gram', '2-gram', 'snowball', 'porter'

#         sep (string): separator character for elements in `x`.

#         transformer (function): function applyed to each group name.

#     Returns:
#         A Thesaurus object.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame({
#     ...    'f': ['a b c a b a',
#     ...          'a b c a b',
#     ...          'a b c a b',
#     ...          'A C b',
#     ...          'a b',
#     ...          'a, b, c, a',
#     ...          'a B'],
#     ... })
#     >>> df
#                  f
#     0  a b c a b a
#     1    a b c a b
#     2    a b c a b
#     3        A C b
#     4          a b
#     5   a, b, c, a
#     6          a B

#     >>> text_clustering(df.f) # doctest: +NORMALIZE_WHITESPACE
#     {
#       "A C b": [
#         "A C b",
#         "a b c a b",
#         "a b c a b a",
#         "a, b, c, a"
#       ],
#       "a B": [
#         "a B",
#         "a b"
#       ]
#     }
#     ignore_case=False, full_match=True, use_re=False, compiled=False


#     >>> text_clustering(df.f, name_strategy='shortest') # doctest: +NORMALIZE_WHITESPACE
#     {
#       "A C b": [
#         "A C b",
#         "a b c a b",
#         "a b c a b a",
#         "a, b, c, a"
#         ],
#       "a b": [
#         "a B",
#         "a b"
#         ]
#     }
#     ignore_case=False, full_match=True, use_re=False, compiled=False

#     >>> text_clustering(df.f, name_strategy='longest') # doctest: +NORMALIZE_WHITESPACE
#     {
#       "a b": [
#         "a B",
#         "a b"
#       ],
#       "a b c a b a": [
#         "A C b",
#         "a b c a b",
#         "a b c a b a",
#         "a, b, c, a"
#       ]
#     }
#     ignore_case=False, full_match=True, use_re=False, compiled=False

#     >>> df = pd.DataFrame({
#     ...    'f': ['a b; c a; b a',
#     ...          'A b; c A; b',
#     ...          'a b; C A; B',
#     ...          'A C; b',
#     ...          None,
#     ...          'a b',
#     ...          'a; b; c; a',
#     ...          'a B'],
#     ... })
#     >>> df
#                    f
#     0  a b; c a; b a
#     1    A b; c A; b
#     2    a b; C A; B
#     3         A C; b
#     4           None
#     5            a b
#     6     a; b; c; a
#     7            a B

#     >>> text_clustering(df.f, sep=';', name_strategy='longest') # doctest: +NORMALIZE_WHITESPACE
#     {
#       "a b": [
#         "A b",
#         "a B",
#         "a b",
#         "b a"
#       ],
#       "b": [
#         "B",
#         "b"
#       ],
#       "c a": [
#         "A C",
#         "C A",
#         "c A",
#         "c a"
#       ]
#     }
#     ignore_case=False, full_match=True, use_re=False, compiled=False


#     """
#     x = x.dropna()
#     if sep is not None:
#         x = x.map(lambda w: w.split(sep))
#         x = x.explode()
#     x = x.map(lambda w: w.strip())
#     x = x.unique()
#     x = pd.DataFrame({"col": x.tolist()})
#     if key == "fingerprint":
#         f = fingerprint
#     elif key == "1-gram":
#         f = one_gram
#     elif key == "2-gram":
#         f = two_gram
#     elif key == "porter":
#         f = stemmer_porter
#     else:
#         f = stemmer_snowball
#     x["key"] = x.col.map(f)
#     grp = x.groupby(by="key").agg({"col": list})
#     grp["listlen"] = grp.col.map(len)
#     grp = grp[grp.listlen.map(lambda w: w > 1)]
#     grp["col"] = grp.col.map(lambda w: pd.Series(w))
#     grp["groupname"] = None
#     if name_strategy is None:
#         name_strategy = "mostfrequent"
#     if name_strategy == "mostfrequent":
#         grp["groupname"] = grp.col.map(
#             lambda w: w.value_counts()[w.value_counts() == w.value_counts().max()]
#             .sort_index()
#             .index[0]
#         )
#     if name_strategy == "longest":
#         grp["groupname"] = grp.col.map(
#             lambda w: sorted(w.tolist(), key=len, reverse=True)[0]
#         )
#     if name_strategy == "shortest":
#         grp["groupname"] = grp.col.map(
#             lambda w: sorted(w.tolist(), key=len, reverse=False)[0]
#         )
#     if transformer is not None:
#         grp["groupname"] = grp.groupname.map(transformer)
#     result = {key: sorted(value.tolist()) for key, value in zip(grp.groupname, grp.col)}
#     return Thesaurus(result, ignore_case=False, full_match=True, use_re=False)


# #


# class Thesaurus:
#     def __init__(self, x={}, ignore_case=True, full_match=False, use_re=False):
#         self._thesaurus = x
#         self._ignore_case = ignore_case
#         self._full_match = full_match
#         self._use_re = use_re
#         self._dict = None
#         self._compiled = None

#     @property
#     def thesaurus(self):
#         return self._thesaurus

#     def __repr__(self):
#         """Returns a json representation of the Thesaurus.
#         """
#         text = json.dumps(self._thesaurus, indent=2, sort_keys=True)
#         text += "\nignore_case={}, full_match={}, use_re={}, compiled={}".format(
#             self._ignore_case.__repr__(),
#             self._full_match.__repr__(),
#             self._use_re.__repr__(),
#             self._compiled is not None,
#         )
#         return text

#     def __str__(self):
#         return self.__repr__()

#     def compile(self):
#         self._compiled = {}
#         for key in self._thesaurus:
#             patterns = self._thesaurus[key]
#             if self._use_re is False:
#                 patterns = [re.escape(pattern) for pattern in patterns]
#             if self._full_match is True:
#                 patterns = ["^" + pattern + "$" for pattern in patterns]
#             if self._ignore_case is True:
#                 patterns = [re.compile(pattern, re.I) for pattern in patterns]
#             else:
#                 patterns = [re.compile(pattern) for pattern in patterns]
#             self._compiled[key] = patterns

#         return self

#     def apply(self, x, sep=None):
#         """Apply a thesaurus to a string x.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> df = pd.DataFrame({
#         ...    'f': ['aaa', 'bbb', 'ccc aaa', 'ccc bbb', 'ddd eee', 'ddd fff',  None, 'zzz'],
#         ... })
#         >>> df # doctest: +NORMALIZE_WHITESPACE
#                  f
#         0      aaa
#         1      bbb
#         2  ccc aaa
#         3  ccc bbb
#         4  ddd eee
#         5  ddd fff
#         6     None
#         7      zzz

#         >>> patterns = {'aaa':['aaa', 'bbb', 'eee', 'fff'],  '1':['000']}
#         >>> thesaurus = Thesaurus(patterns)
#         >>> thesaurus = thesaurus.compile()
#         >>> thesaurus
#         {
#           "1": [
#             "000"
#           ],
#           "aaa": [
#             "aaa",
#             "bbb",
#             "eee",
#             "fff"
#           ]
#         }
#         ignore_case=True, full_match=False, use_re=False, compiled=True

#         >>> df.f.map(lambda x: thesaurus.apply(x))
#         0     aaa
#         1     aaa
#         2     aaa
#         3     aaa
#         4     aaa
#         5     aaa
#         6    None
#         7     zzz
#         Name: f, dtype: object

#         >>> df = pd.DataFrame({
#         ...    'f': ['aaa|ccc aaa', 'bbb|ccc bbb', 'ccc aaa', 'ccc bbb', 'ddd eee', 'ddd fff',  None, 'zzz'],
#         ... })
#         >>> df # doctest: +NORMALIZE_WHITESPACE
#                      f
#         0  aaa|ccc aaa
#         1  bbb|ccc bbb
#         2      ccc aaa
#         3      ccc bbb
#         4      ddd eee
#         5      ddd fff
#         6         None
#         7          zzz
#         >>> df.f.map(lambda x: thesaurus.apply(x, sep='|'))
#         0    aaa|aaa
#         1    aaa|aaa
#         2        aaa
#         3        aaa
#         4        aaa
#         5        aaa
#         6       None
#         7        zzz
#         Name: f, dtype: object

#         >>> import pandas as pd
#         >>> df = pd.DataFrame({
#         ...    'f': ['0', '1', '2', '3', None, '4', '5', '6', '7', '8', '9'],
#         ... })
#         >>> df # doctest: +NORMALIZE_WHITESPACE
#                f
#         0      0
#         1      1
#         2      2
#         3      3
#         4   None
#         5      4
#         6      5
#         7      6
#         8      7
#         9      8
#         10     9

#         >>> patterns = {
#         ...     'a':['0', '1', '2'],
#         ...     'b':['4', '5', '6'],
#         ...     'c':['7', '8', '9']
#         ... }
#         >>> thesaurus = Thesaurus(patterns, ignore_case=False, full_match=True)
#         >>> thesaurus = thesaurus.compile()
#         >>> df.f.map(lambda x: thesaurus.apply(x)) # doctest: +NORMALIZE_WHITESPACE
#         0        a
#         1        a
#         2        a
#         3        3
#         4     None
#         5        b
#         6        b
#         7        b
#         8        c
#         9        c
#         10       c
#         Name: f, dtype: object


#         >>> df = pd.DataFrame({
#         ...    'f': ['a b, A B', 'A b, A B', None, 'b c', 'b, B A', 'b, a, c', 'A, B'],
#         ... })
#         >>> df
#                   f
#         0  a b, A B
#         1  A b, A B
#         2      None
#         3       b c
#         4    b, B A
#         5   b, a, c
#         6      A, B
#         >>> patterns = {
#         ...     '0':['a b', 'A B', 'B A'],
#         ...     '1':['b c'],
#         ...     '2':['a', 'b']
#         ... }
#         >>> thesaurus = Thesaurus(patterns, ignore_case=False, full_match=True)
#         >>> thesaurus = thesaurus.compile()
#         >>> df.f.map(lambda x: thesaurus.apply(x, sep=','))
#         0      0,0
#         1    A b,0
#         2     None
#         3        1
#         4      2,0
#         5    2,2,c
#         6      A,B
#         Name: f, dtype: object


#         """

#         def f(z):
#             z = z.strip()
#             for key in self._compiled:
#                 for pattern in self._compiled[key]:
#                     if len(pattern.findall(z)):
#                         return key
#             return z

#         #
#         # Main body
#         #
#         if pd.isna(x):
#             return x
#         if sep is None:
#             x = [x]
#         else:
#             x = x.split(sep)
#         result = [f(z) for z in x]
#         if sep is None:
#             return result[0]
#         return sep.join(result)

#     def find_and_replace(self, x, sep=None):
#         """Applies a thesaurus to a string, reemplacing the portion of string
#         matching the current pattern with the key.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame({
#         ...    'f': ['AAA', 'BBB', 'ccc AAA', 'ccc BBB', 'ddd EEE', 'ddd FFF',  None, 'zzz'],
#         ... })
#         >>> df # doctest: +NORMALIZE_WHITESPACE
#                  f
#         0      AAA
#         1      BBB
#         2  ccc AAA
#         3  ccc BBB
#         4  ddd EEE
#         5  ddd FFF
#         6     None
#         7      zzz
#         >>> patterns = {'aaa':['AAA', 'BBB', 'EEE', 'FFF'],  '1':['000']}
#         >>> thesaurus = Thesaurus(patterns)
#         >>> thesaurus = thesaurus.compile()
#         >>> df.f.map(lambda x: thesaurus.find_and_replace(x))
#         0        aaa
#         1        aaa
#         2    ccc aaa
#         3    ccc aaa
#         4    ddd aaa
#         5    ddd aaa
#         6       None
#         7        zzz
#         Name: f, dtype: object

#         >>> df = pd.DataFrame({
#         ...    'f': ['AAA|ccc AAA', 'BBB ccc|ccc', 'ccc AAA', 'ccc BBB', 'ddd EEE', 'ddd FFF',  None, 'zzz'],
#         ... })
#         >>> df # doctest: +NORMALIZE_WHITESPACE
#                      f
#         0  AAA|ccc AAA
#         1  BBB ccc|ccc
#         2      ccc AAA
#         3      ccc BBB
#         4      ddd EEE
#         5      ddd FFF
#         6         None
#         7          zzz
#         >>> df.f.map(lambda x: thesaurus.find_and_replace(x, sep='|'))
#         0    aaa|ccc aaa
#         1    aaa ccc|ccc
#         2        ccc aaa
#         3        ccc aaa
#         4        ddd aaa
#         5        ddd aaa
#         6           None
#         7            zzz
#         Name: f, dtype: object
#         """

#         def f(z):
#             z = z.strip()
#             for key in self._compiled:
#                 for pattern in self._compiled[key]:
#                     w = pattern.sub(key, z)
#                     if z != w:
#                         return w
#             return z

#         #
#         # Main body
#         #
#         if x is None:
#             return None
#         if sep is None:
#             x = [x]
#         else:
#             x = x.split(sep)
#         result = [f(z) for z in x]
#         if sep is None:
#             return result[0]
#         return sep.join(result)

#     def merge_keys(self, key, popkey):
#         """Adds the strings associated to popkey to key and delete popkey.
#         """
#         if isinstance(popkey, list):
#             for k in popkey:
#                 self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[k]
#                 self._thesaurus.pop(k)
#         else:
#             self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[popkey]
#             self._thesaurus.pop(popkey)

#     def pop_key(self, key):
#         """Deletes key from thesaurus.
#         """
#         self._thesaurus.pop(key)

#     def change_key(self, current_key, new_key):
#         self._thesaurus[new_key] = self._thesaurus[current_key]
#         self._thesaurus.popkey(current_key)

#     def to_dict(self):
#         result = {}
#         for key in self._thesaurus.keys():
#             for value in self._thesaurus[key]:
#                 result[value] = key
#         return result

#     def apply_as_dict(self, x, sep=None):

#         if x is None:
#             return None

#         if sep is None:
#             x = [x]
#         else:
#             x = x.split(sep)

#         result = [self._dict[z] if z in self._dict.keys() else z for z in x]

#         if sep is None:
#             return result[0]

#         return sep.join(result)


# #  def text_nesting(
# #     x, search_strategy="fingerprint", sep=None, transformer=None, max_distance=None
# # ):
# #     """

# #     Examples
# #     ----------------------------------------------------------------------------------------------

# #     >>> import pandas as pd
# #     >>> df = pd.DataFrame({
# #     ...    'f': ['a',
# #     ...          'a b',
# #     ...          'a b c',
# #     ...          'a b c d',
# #     ...          'a e',
# #     ...          'a f',
# #     ...          'a b e',
# #     ...          'a b e f',
# #     ...          'a b e f g'],
# #     ... })
# #     >>> df # doctest: +NORMALIZE_WHITESPACE
# #                f
# #     0          a
# #     1        a b
# #     2      a b c
# #     3    a b c d
# #     4        a e
# #     5        a f
# #     6      a b e
# #     7    a b e f
# #     8  a b e f g

# #     >>> text_nesting(df.f, sep=',') # doctest: +NORMALIZE_WHITESPACE
# #     {
# #       "a": [
# #         "a",
# #         "a e",
# #         "a f"
# #       ],
# #       "a b": [
# #         "a b",
# #         "a b e"
# #       ],
# #       "a b c": [
# #         "a b c",
# #         "a b c d"
# #       ],
# #       "a b e f": [
# #         "a b e f",
# #         "a b e f g"
# #       ]
# #     }
# #     ignore_case=False, full_match=True, use_re=False, compiled=False

# #     >>> df = pd.DataFrame({
# #     ...    'f': ['neural networks; Artificial Neural Networks']
# #     ... })
# #     >>> df # doctest: +NORMALIZE_WHITESPACE
# #                                                  f
# #     0  neural networks; Artificial Neural Networks
# #     >>> text_nesting(df.f, sep=';', max_distance=1) # doctest: +NORMALIZE_WHITESPACE
# #     {
# #       "neural networks": [
# #         "Artificial Neural Networks",
# #         "neural networks"
# #       ]
# #     }
# #     ignore_case=False, full_match=True, use_re=False, compiled=False

# #     """

# #     x = x.dropna()

# #     if sep is not None:
# #         x = pd.Series([z.strip() for y in x for z in y.split(sep)])

# #     result = {}
# #     selected = {text: False for text in x.tolist()}

# #     max_text_len = max([len(text) for text in x])
# #     sorted_x = []
# #     for text_len in range(max_text_len, -1, -1):
# #         texts = x[[True if len(w) == text_len else False for w in x]]
# #         texts = sorted(texts)
# #         sorted_x += texts
# #     x = sorted_x

# #     # for pattern in x:
# #     for iter in tqdm(range(len(x))):

# #         pattern = x[iter]

# #         if pattern == "":
# #             continue

# #         if selected[pattern] is True:
# #             continue

# #         nested_texts = [
# #             text
# #             for text in x
# #             if selected[text] is False and steamming_all(pattern, text)
# #         ]

# #         if max_distance is not None:
# #             nested_texts = [
# #                 z
# #                 for z in nested_texts
# #                 if abs(len(pattern.split()) - len(z.split())) <= max_distance
# #             ]

# #         if len(nested_texts) > 1:
# #             nested_texts = sorted(list(set(nested_texts)))

# #         if len(nested_texts) > 1:

# #             if transformer is not None:
# #                 pattern = transformer(pattern)
# #             if pattern in result.keys():
# #                 result[pattern] += nested_texts
# #             else:
# #                 result[pattern] = nested_texts
# #             for txt in nested_texts:
# #                 selected[txt] = True

# #     return Thesaurus(result, ignore_case=False, full_match=True, use_re=False)


if __name__ == "__main__":
    #
    import doctest

    doctest.testmod()
