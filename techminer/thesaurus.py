"""
Thesaurus
==================================================================================================

"""
import pandas as pd
import json
from techminer.strings import find_string, replace_string, fingerprint, steamming_all


def text_clustering(
    x,
    name_strategy="mostfrequent",
    search_strategy="fingerprint",
    sep=None,
    transformer=None,
    min_cluster_size=1,
):
    """Builds a thesaurus by clustering a list of strings.

    Args:
        x (list): list  of string to create thesaurus.

        name_strategy (string): method for assigning keys in thesaurus.

            * 'mostfrequent': Most frequent string in the cluster.

            * 'longest': Longest string in the cluster.

            * 'shortest': Shortest string in the cluster.

        search_strategy (string): cluster method.

            * 'fingerprint'.

        sep (string): separator character for elements in `x`.

    Returns:
        A Thesaurus object.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['a b c a b a',
    ...          'a b c a b',
    ...          'a b c a b',
    ...          'A C b',
    ...          'a b',
    ...          'a, b, c, a',
    ...          'a B'],
    ... })
    >>> df
                 f
    0  a b c a b a
    1    a b c a b
    2    a b c a b
    3        A C b
    4          a b
    5   a, b, c, a
    6          a B

    >>> text_clustering(df.f) # doctest: +NORMALIZE_WHITESPACE
    {
      "a b": [
        "a B",
        "a b"
        ],
      "a b c a b": [
        "A C b",
        "a b c a b",
        "a b c a b a",
        "a, b, c, a"
        ]
    }

    >>> text_clustering(df.f, name_strategy='shortest') # doctest: +NORMALIZE_WHITESPACE
    {
      "A C b": [
        "A C b",
        "a b c a b",
        "a b c a b a",
        "a, b, c, a"
        ],
      "a b": [
        "a B",
        "a b"
        ]
    }

    >>> text_clustering(df.f, name_strategy='longest') # doctest: +NORMALIZE_WHITESPACE
    {
      "a B": [
        "a B",
        "a b"
        ],
      "a b c a b a": [
        "A C b",
        "a b c a b",
        "a b c a b a",
        "a, b, c, a"
        ]
    }

    >>> df = pd.DataFrame({
    ...    'f': ['a b, c a, b a',
    ...          'A b, c A, b',
    ...          'a b, C A, B',
    ...          'A C, b',
    ...          None,
    ...          'a b',
    ...          'a, b, c, a',
    ...          'a B'],
    ... })
    >>> df
                   f
    0  a b, c a, b a
    1    A b, c A, b
    2    a b, C A, B
    3         A C, b
    4           None
    5            a b
    6     a, b, c, a
    7            a B

    >>> text_clustering(df.f, sep=',', name_strategy='longest') # doctest: +NORMALIZE_WHITESPACE
    {
      "A C": [
        "A C",
        "C A",
        "c A",
        "c a"
      ],
      "a B": [
        "A b",
        "a B",
        "a b",
        "b a"
      ],
      "b": [
        "B",
        "b"
      ]
    }


    """

    x = x.dropna()

    if sep is not None:
        x = pd.Series([z.strip() for y in x for z in y.split(sep)])

    if search_strategy == "fingerprint":
        y = x.map(lambda w: fingerprint(w))
    y = y.sort_values()
    counts = y.value_counts()
    counts = counts[counts > 1]

    result = {}
    for z in counts.index.tolist():

        w = x[y == z]

        if name_strategy is None or name_strategy == "mostfrequent":
            m = w.value_counts().sort_values()
            m = m[m == m[-1]].sort_index()
            groupName = m.index[-1]

        if name_strategy == "longest" or name_strategy == "shortest":
            m = pd.Series([len(a) for a in w], index=w.index).sort_values()
            if name_strategy == "longest":
                groupName = w[m.index[-1]]
            else:
                groupName = w[m.index[0]]

        if transformer is not None:
            groupName = transformer(groupName)

        z = w.sort_values().unique().tolist()
        if len(z) > min_cluster_size:
            if groupName in result.keys():
                result[groupName] += z
            else:
                result[groupName] = z

    return Thesaurus(result, ignore_case=False, full_match=True, use_re=False)


def text_nesting(
    x, search_strategy="fingerprint", sep=None, transformer=None, max_distance=None
):
    """

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['a',
    ...          'a b',
    ...          'a b c',
    ...          'a b c d',
    ...          'a e',
    ...          'a f',
    ...          'a b e',
    ...          'a b e f',
    ...          'a b e f g'],
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
               f
    0          a
    1        a b
    2      a b c
    3    a b c d
    4        a e
    5        a f
    6      a b e
    7    a b e f
    8  a b e f g

    >>> text_nesting(df.f, sep=',') # doctest: +NORMALIZE_WHITESPACE
    {
      "a": [
        "a",
        "a e",
        "a f"
      ],
      "a b": [
        "a b",
        "a b e"
      ],
      "a b c": [
        "a b c",
        "a b c d"
      ],
      "a b e f": [
        "a b e f",
        "a b e f g"
      ]
    }

    >>> df = pd.DataFrame({
    ...    'f': ['neural networks; Artificial Neural Networks']
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
                                                 f
    0  neural networks; Artificial Neural Networks
    >>> text_nesting(df.f, sep=';', max_distance=1) # doctest: +NORMALIZE_WHITESPACE
    {
      "neural networks": [
        "Artificial Neural Networks",
        "neural networks"
      ]
    }
    """

    x = x.dropna()

    if sep is not None:
        x = pd.Series([z.strip() for y in x for z in y.split(sep)])

    result = {}
    selected = {text: False for text in x.tolist()}

    max_text_len = max([len(text) for text in x])
    sorted_x = []
    for text_len in range(max_text_len, -1, -1):
        texts = x[[True if len(w) == text_len else False for w in x]]
        texts = sorted(texts)
        sorted_x += texts
    x = sorted_x

    for pattern in x:

        if pattern == "":
            continue

        if selected[pattern] is True:
            continue

        nested_texts = [
            text
            for text in x
            if selected[text] is False and steamming_all(pattern, text)
        ]

        if max_distance is not None:
            nested_texts = [
                z
                for z in nested_texts
                if abs(len(pattern.split()) - len(z.split())) <= max_distance
            ]

        if len(nested_texts) > 1:
            nested_texts = sorted(list(set(nested_texts)))

        if len(nested_texts) > 1:

            if transformer is not None:
                pattern = transformer(pattern)
            if pattern in result.keys():
                result[pattern] += nested_texts
            else:
                result[pattern] = nested_texts
            for txt in nested_texts:
                selected[txt] = True

    return Thesaurus(result, ignore_case=False, full_match=True, use_re=False)


class Thesaurus:
    def __init__(self, x={}, ignore_case=True, full_match=False, use_re=False):
        self._thesaurus = x
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._use_re = use_re
        self._dict = None

    @property
    def thesaurus(self):
        return self._thesaurus

    def apply(self, x, sep=None):
        """Apply a thesaurus to a string x.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> df = pd.DataFrame({
        ...    'f': ['aaa', 'bbb', 'ccc aaa', 'ccc bbb', 'ddd eee', 'ddd fff',  None, 'zzz'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
                 f
        0      aaa
        1      bbb
        2  ccc aaa
        3  ccc bbb
        4  ddd eee
        5  ddd fff
        6     None
        7      zzz

        >>> d = {'aaa':['aaa', 'bbb', 'eee', 'fff'],  '1':['000']}
        >>> df.f.map(lambda x: Thesaurus(d).apply(x))
        0     aaa
        1     aaa
        2     aaa
        3     aaa
        4     aaa
        5     aaa
        6    None
        7     zzz
        Name: f, dtype: object

        >>> df = pd.DataFrame({
        ...    'f': ['aaa|ccc aaa', 'bbb|ccc bbb', 'ccc aaa', 'ccc bbb', 'ddd eee', 'ddd fff',  None, 'zzz'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
                     f
        0  aaa|ccc aaa
        1  bbb|ccc bbb
        2      ccc aaa
        3      ccc bbb
        4      ddd eee
        5      ddd fff
        6         None
        7          zzz
        >>> df.f.map(lambda x: Thesaurus(d).apply(x, sep='|'))
        0    aaa|aaa
        1    aaa|aaa
        2        aaa
        3        aaa
        4        aaa
        5        aaa
        6       None
        7        zzz
        Name: f, dtype: object

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...    'f': ['0', '1', '2', '3', None, '4', '5', '6', '7', '8', '9'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
               f
        0      0
        1      1
        2      2
        3      3
        4   None
        5      4
        6      5
        7      6
        8      7
        9      8
        10     9

        >>> d = {'a':['0', '1', '2'],
        ...      'b':['4', '5', '6'],
        ...      'c':['7', '8', '9']}

        >>> df.f.map(lambda x: Thesaurus(d, ignore_case=False, full_match=True).apply(x)) # doctest: +NORMALIZE_WHITESPACE
        0        a
        1        a
        2        a
        3        3
        4     None
        5        b
        6        b
        7        b
        8        c
        9        c
        10       c
        Name: f, dtype: object


        >>> df = pd.DataFrame({
        ...    'f': ['a b, A B', 'A b, A B', None, 'b c', 'b, B A', 'b, a, c', 'A, B'],
        ... })
        >>> df
                  f
        0  a b, A B
        1  A b, A B
        2      None
        3       b c
        4    b, B A
        5   b, a, c
        6      A, B
        >>> d = {'0':['a b', 'A B', 'B A'],
        ...      '1':['b c'],
        ...      '2':['a', 'b']}
        >>> df.f.map(lambda x: Thesaurus(d, ignore_case=False, full_match=True).apply(x, sep=','))
        0      0,0
        1    A b,0
        2     None
        3        1
        4      2,0
        5    2,2,c
        6      A,B
        Name: f, dtype: object


        """

        def _apply(z):
            """Transform the string z using the thesaurus. Returns when there is a match.
            """

            z = z.strip()

            for key in self._thesaurus.keys():
                for pattern in self._thesaurus[key]:

                    y = find_string(
                        pattern=pattern,
                        x=z,
                        ignore_case=self._ignore_case,
                        full_match=self._full_match,
                        use_re=self._use_re,
                    )

                    if y is not None:
                        return key
            return z

        ##
        ## main body
        ##

        if x is None:
            return None

        if sep is None:
            x = [x]
        else:
            x = x.split(sep)

        result = [_apply(z) for z in x]

        if sep is None:
            return result[0]

        return sep.join(result)

    def find_and_replace(self, x, sep=None):
        """Applies a thesaurus to a string, reemplacing the portion of string
        matching the current pattern with the key.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> df = pd.DataFrame({
        ...    'f': ['AAA', 'BBB', 'ccc AAA', 'ccc BBB', 'ddd EEE', 'ddd FFF',  None, 'zzz'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
                 f
        0      AAA
        1      BBB
        2  ccc AAA
        3  ccc BBB
        4  ddd EEE
        5  ddd FFF
        6     None
        7      zzz
        >>> d = {'aaa':['AAA', 'BBB', 'EEE', 'FFF'],  '1':['000']}
        >>> df.f.map(lambda x: Thesaurus(d).find_and_replace(x))
        0        aaa
        1        aaa
        2    ccc aaa
        3    ccc aaa
        4    ddd aaa
        5    ddd aaa
        6       None
        7        zzz
        Name: f, dtype: object

        >>> df = pd.DataFrame({
        ...    'f': ['AAA|ccc AAA', 'BBB ccc|ccc', 'ccc AAA', 'ccc BBB', 'ddd EEE', 'ddd FFF',  None, 'zzz'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
                     f
        0  AAA|ccc AAA
        1  BBB ccc|ccc
        2      ccc AAA
        3      ccc BBB
        4      ddd EEE
        5      ddd FFF
        6         None
        7          zzz
        >>> df.f.map(lambda x: Thesaurus(d).find_and_replace(x, sep='|'))
        0    aaa|ccc aaa
        1    aaa ccc|ccc
        2        ccc aaa
        3        ccc aaa
        4        ddd aaa
        5        ddd aaa
        6           None
        7            zzz
        Name: f, dtype: object
        """

        def _apply_and_replace(z):

            z = z.strip()

            for key in self._thesaurus.keys():

                for pattern in self._thesaurus[key]:

                    w = replace_string(
                        pattern=pattern,
                        x=z,
                        repl=key,
                        ignore_case=self._ignore_case,
                        full_match=self._full_match,
                        use_re=self._use_re,
                    )

                    if z != w:
                        return w

            return z

        if x is None:
            return None

        if sep is None:
            x = [x]
        else:
            x = x.split(sep)

        result = [_apply_and_replace(z) for z in x]

        if sep is None:
            return result[0]

        return sep.join(result)

    def __repr__(self):
        """Returns a json representation of the Thesaurus.
        """
        return json.dumps(self._thesaurus, indent=2, sort_keys=True)

    def __str__(self):
        return self.__repr__()

    def merge_keys(self, key, popkey):
        """Adds the strings associated to popkey to key and delete popkey.
        """
        if isinstance(popkey, list):
            for k in popkey:
                self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[k]
                self._thesaurus.pop(k)
        else:
            self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[popkey]
            self._thesaurus.pop(popkey)

    def pop_key(self, key):
        """Deletes key from thesaurus.
        """
        self._thesaurus.pop(key)

    def change_key(self, current_key, new_key):
        self._thesaurus[new_key] = self._thesaurus[current_key]
        self._thesaurus.popkey(current_key)

    def to_dict(self):
        result = {}
        for key in self._thesaurus.keys():
            for value in self._thesaurus[key]:
                result[value] = key
        return result

    def compile(self):
        self._dict = self.to_dict()

    def appy_as_dict(self, x, sep=None):

        if x is None:
            return None

        if sep is None:
            x = [x]
        else:
            x = x.split(sep)

        result = [self._dict[z] if z in self._dict.keys() else z for z in x]

        if sep is None:
            return result[0]

        return sep.join(result)
