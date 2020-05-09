"""
TechMiner.DataFrame
==================================================================================================




"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .strings import remove_accents

SCOPUS_SEPS = {
    "Authors": ",",
    "Author(s) ID": ";",
    "Author Keywords": ";",
    "Index Keywords": ";",
    "ID": ";",
}


def heatmap(df, cmap):
    """Display the dataframe as a heatmap in Jupyter.

        Args:
            df (pandas.DataFrame): dataframe as a matrix.
            cmap (colormap): colormap used to build a heatmap.

        Returns:
            DataFrame. Exploded dataframe.


    """
    return df.style.backgroud_gradient(cmap=cmap)


def sort_by_numdocuments(
    df, matrix, axis=0, ascending=True, kind="quicksort", axis_name=None, axis_sep=None
):
    """Sorts a matrix axis by the number of documents.


    Args:
        df (pandas.DataFrame): dataframe with bibliographic information.
        matrix (pandas.DataFrame): matrix to sort.
        axis ({0 or ‘index’, 1 or ‘columns’}), default 0: axis to be sorted.
        ascending (bool): sort ascending?.
        kind (str): ‘quicksort’, ‘mergesort’, ‘heapsort’.
        axis_name (str): column name used to sort by number of documents.
        axis_sep (str): character used to separate the internal values of column axis_name.

    Returns:
        DataFrame sorted.

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "c0": ["D"] * 4 + ["B"] * 3 + ["C"] * 2 + ["A"],
    ...         "c1": ["a"] * 4 + ["c"] * 3 + ["b"] * 2 + ["d"],
    ...         "Cited by": list(range(10)),
    ...         "ID": list(range(10)),
    ...     },
    ... )
    >>> df
      c0 c1  Cited by  ID
    0  D  a         0   0
    1  D  a         1   1
    2  D  a         2   2
    3  D  a         3   3
    4  B  c         4   4
    5  B  c         5   5
    6  B  c         6   6
    7  C  b         7   7
    8  C  b         8   8
    9  A  d         9   9

    >>> matrix = pd.DataFrame(
    ...     {"D": [0, 1, 2, 3], "B": [4, 5, 6, 7], "A": [8, 9, 10, 11], "C": [12, 13, 14, 15],},
    ...     index=list("badc"),
    ... )
    >>> matrix
       D  B   A   C
    b  0  4   8  12
    a  1  5   9  13
    d  2  6  10  14
    c  3  7  11  15

    >>> sort_by_numdocuments(df, matrix, axis='columns', ascending=True, axis_name='c0')
        A  B   C  D
    b   8  4  12  0
    a   9  5  13  1
    d  10  6  14  2
    c  11  7  15  3

    >>> sort_by_numdocuments(df, matrix, axis='columns', ascending=False, axis_name='c0')
       D   C  B   A
    b  0  12  4   8
    a  1  13  5   9
    d  2  14  6  10
    c  3  15  7  11

    >>> sort_by_numdocuments(df, matrix, axis='index', ascending=True, axis_name='c1')
       D  B   A   C
    a  1  5   9  13
    b  0  4   8  12
    c  3  7  11  15
    d  2  6  10  14

    >>> sort_by_numdocuments(df, matrix, axis='index', ascending=False, axis_name='c1')
       D  B   A   C
    d  2  6  10  14
    c  3  7  11  15
    b  0  4   8  12
    a  1  5   9  13

    """
    terms = DataFrame(df).documents_by_term(column=axis_name, sep=axis_sep)
    terms_sorted = (
        terms.sort_values(by=axis_name, kind=kind, ascending=ascending)
        .iloc[:, 0]
        .tolist()
    )
    if axis == "index":
        return matrix.loc[terms_sorted, :]
    return matrix.loc[:, terms_sorted]


def sort_by_citations(
    df, matrix, axis=0, ascending=True, kind="quicksort", axis_name=None, axis_sep=None
):
    """Sorts a matrix axis by the citations.


    Args:
        df (pandas.DataFrame): dataframe with bibliographic information.
        matrix (pandas.DataFrame): matrix to sort.
        axis ({0 or ‘index’, 1 or ‘columns’}), default 0: axis to be sorted.
        ascending (bool): sort ascending?.
        kind (str): ‘quicksort’, ‘mergesort’, ‘heapsort’.
        axis_name (str): column name used to sort by citations.
        axis_sep (str): character used to separate the internal values of column axis_name.

    Returns:
        DataFrame sorted.

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "c0": ["D"] * 4 + ["B"] * 3 + ["C"] * 2 + ["A"],
    ...         "c1": ["a"] * 4 + ["c"] * 3 + ["b"] * 2 + ["d"],
    ...         "Cited by": list(range(10)),
    ...         "ID": list(range(10)),
    ...     },
    ... )
    >>> df
      c0 c1  Cited by  ID
    0  D  a         0   0
    1  D  a         1   1
    2  D  a         2   2
    3  D  a         3   3
    4  B  c         4   4
    5  B  c         5   5
    6  B  c         6   6
    7  C  b         7   7
    8  C  b         8   8
    9  A  d         9   9

    >>> matrix = pd.DataFrame(
    ...     {"D": [0, 1, 2, 3], "B": [4, 5, 6, 7], "A": [8, 9, 10, 11], "C": [12, 13, 14, 15],},
    ...     index=list("badc"),
    ... )
    >>> matrix
       D  B   A   C
    b  0  4   8  12
    a  1  5   9  13
    d  2  6  10  14
    c  3  7  11  15

    >>> sort_by_citations(df, matrix, axis='columns', ascending=True, axis_name='c0')
        A  B   C  D
    b   8  4  12  0
    a   9  5  13  1
    d  10  6  14  2
    c  11  7  15  3

    >>> sort_by_citations(df, matrix, axis='columns', ascending=False, axis_name='c0')
       D   C  B   A
    b  0  12  4   8
    a  1  13  5   9
    d  2  14  6  10
    c  3  15  7  11

    >>> sort_by_citations(df, matrix, axis='index', ascending=True, axis_name='c1')
       D  B   A   C
    a  1  5   9  13
    b  0  4   8  12
    c  3  7  11  15
    d  2  6  10  14

    >>> sort_by_citations(df, matrix, axis='index', ascending=False, axis_name='c1')
       D  B   A   C
    d  2  6  10  14
    c  3  7  11  15
    b  0  4   8  12
    a  1  5   9  13

    """
    terms = DataFrame(df).citations_by_term(column=axis_name, sep=axis_sep)
    terms_sorted = (
        terms.sort_values(by=axis_name, kind=kind, ascending=ascending)
        .iloc[:, 0]
        .tolist()
    )
    if axis == "index":
        return matrix.loc[terms_sorted, :]
    return matrix.loc[:, terms_sorted]


class DataFrame(pd.DataFrame):
    """Data structure derived from a `pandas:DataFrame 
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame>`_. 
    object with specialized functions to summarize and query bibliographic data.


    """

    #
    # Compatitbility with pandas.DataFrame
    #

    @property
    def _constructor_expanddim(self):
        return self

    def explode(self, column, sep=None):
        """Transform each element of a field to a row, reseting index values.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame. Exploded dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...   {
        ...     "Authors": "author 0,author 1,author 2;author 3;author 4".split(";"),
        ...     "ID": list(range(3)),
        ...   }
        ... )
        >>> df
                              Authors  ID
        0  author 0,author 1,author 2   0
        1                    author 3   1
        2                    author 4   2

        >>> DataFrame(df).explode('Authors')
            Authors  ID
        0  author 0   0
        1  author 1   0
        2  author 2   0
        3  author 3   1
        4  author 4   2

        """

        result = self.copy()
        if sep is None and column in SCOPUS_SEPS.keys():
            sep = SCOPUS_SEPS[column]
        if sep is not None:
            result[column] = result[column].map(
                lambda x: sorted(list(set(x.split(sep)))) if isinstance(x, str) else x
            )
            result = result.explode(column)
            result[column] = result[column].map(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            result = result.reset_index(drop=True)

        return result

    #
    # Document ID
    #

    def generate_ID(self, fmt=None):
        """Generates a sequence of integers as ID for each document.


        Args:
            fmt (str): Format used to generate ID column.

        Returns:
            DataFrame.
       
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...   {
        ...     "Authors": "author 0,author 1,author 2;author 3;author 4".split(";"),
        ...   }
        ... )
        >>> df
                              Authors
        0  author 0,author 1,author 2
        1                    author 3
        2                    author 4

        >>> DataFrame(df).generate_ID()
                              Authors  ID
        0  author 0,author 1,author 2   0
        1                    author 3   1
        2                    author 4   2

        """
        if fmt is None:
            self["ID"] = [x for x in range(len(self))]
        else:
            self["ID"] = [fmt.format(x) for x in range(len(self))]
        return DataFrame(self)

    #
    # Distinc authors with same name
    #
    def disambiguate_authors(
        self,
        col_Authors="Authors",
        sep_Authors=None,
        col_AuthorsID="Author(s) ID",
        sep_AuthorsID=None,
    ):
        """Verify if author's names are unique. For duplicated names, based on `Author(s) ID` column, 
        adds a consecutive number to the name.


        Args:
            col_Authors (str): Author's name column.
            sep_Authors (str): Character used as internal separator for the elements in the column with the author's name.
            col_AuthorsID (str): Author's ID column.
            sep_AuthorsID (str): Character used as internal separator for the elements in the column with the author's ID.

        Returns:
            DataFrame.
       

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...   {
        ...      "Authors": "author 0,author 0,author 0;author 0;author 0".split(";"),
        ...      "Author(s) ID": "0;1;2;,3;,4;".split(','),
        ...   }
        ... )
        >>> df
                              Authors Author(s) ID
        0  author 0,author 0,author 0       0;1;2;
        1                    author 0           3;
        2                    author 0           4;

        >>> DataFrame(df).disambiguate_authors()
                                    Authors Author(s) ID
        0  author 0,author 0(1),author 0(2)        0;1;2
        1                       author 0(3)            3
        2                       author 0(4)            4


        """

        if sep_Authors is None:
            sep_Authors = SCOPUS_SEPS[col_Authors]

        self[col_Authors] = self[col_Authors].map(
            lambda x: x[:-1] if x is not None and x[-1] == sep_Authors else x
        )

        if sep_AuthorsID is None:
            sep_AuthorsID = SCOPUS_SEPS[col_AuthorsID]

        self[col_AuthorsID] = self[col_AuthorsID].map(
            lambda x: x[:-1] if x is not None and x[-1] == sep_AuthorsID else x
        )

        data = self[[col_Authors, col_AuthorsID]]
        data = data.dropna()

        data["*info*"] = [
            (a, b) for (a, b) in zip(data[col_Authors], data[col_AuthorsID])
        ]

        data["*info*"] = data["*info*"].map(
            lambda x: [
                (u.strip(), v.strip())
                for u, v in zip(x[0].split(sep_Authors), x[1].split(sep_AuthorsID))
            ]
        )

        data = data[["*info*"]].explode("*info*")
        data = data.reset_index(drop=True)

        names_ids = {}
        for idx in range(len(data)):

            author_name = data.at[idx, "*info*"][0]
            author_id = data.at[idx, "*info*"][1]

            if author_name in names_ids.keys():

                if author_id not in names_ids[author_name]:
                    names_ids[author_name] = names_ids[author_name] + [author_id]
            else:
                names_ids[author_name] = [author_id]

        ids_names = {}
        for author_name in names_ids.keys():
            suffix = 0
            for author_id in names_ids[author_name]:
                if suffix > 0:
                    ids_names[author_id] = author_name + "(" + str(suffix) + ")"
                else:
                    ids_names[author_id] = author_name
                suffix += 1

        result = self.copy()

        result[col_Authors] = result[col_AuthorsID].map(
            lambda x: sep_Authors.join(
                [ids_names[w.strip()] for w in x.split(sep_AuthorsID)]
            )
            if x is not None
            else x
        )

        return DataFrame(result)

    #
    # Accents
    #
    def remove_accents(self):
        """Remove accents for all strings on a DataFrame

        Returns:
            A dataframe.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...   {
        ...      "Authors": "áàÁÀ;éèÉÈ;íìÌÍ;ÑÑÑñññ".split(";"),
        ...   }
        ... )
        >>> DataFrame(df).remove_accents()
          Authors
        0    aaAA
        1    eeEE
        2    iiII
        3  NNNnnn


        """
        return DataFrame(
            self.applymap(lambda x: remove_accents(x) if isinstance(x, str) else x)
        )

    #
    # Basic info
    #

    def coverage(self):
        """Reports the number of not `None` elements for column in a dataframe.

        Returns:
            Pandas DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------


        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...   {
        ...      "Authors": "author 0,author 0,author 0;author 0;author 0;author 1;author 2;author 0".split(";"),
        ...      "Author(s) ID": "0;1;2;,3;,4;,5;,6;,7:".split(','),
        ...      "ID": list(range(6)),
        ...      "Source title": ['source {:d}'.format(w) for w in range(5)] + [None],
        ...      "None field": [None] * 6,
        ...   }
        ... )
        >>> df
                              Authors Author(s) ID  ID Source title None field
        0  author 0,author 0,author 0       0;1;2;   0     source 0       None
        1                    author 0           3;   1     source 1       None
        2                    author 0           4;   2     source 2       None
        3                    author 1           5;   3     source 3       None
        4                    author 2           6;   4     source 4       None
        5                    author 0           7:   5         None       None

        >>> DataFrame(df).coverage()
                 Column  Number of items Coverage (%)
        0       Authors                6      100.00%
        1  Author(s) ID                6      100.00%
        2            ID                6      100.00%
        3  Source title                5       83.33%
        4    None field                0        0.00%

        """

        return pd.DataFrame(
            {
                "Column": self.columns,
                "Number of items": [
                    len(self) - self[col].isnull().sum() for col in self.columns
                ],
                "Coverage (%)": [
                    "{:5.2%}".format((len(self) - self[col].isnull().sum()) / len(self))
                    for col in self.columns
                ],
            }
        )

    #
    #
    # Term extraction
    #
    #

    def extract_terms(self, column, sep=None):
        """Extracts unique terms in a column, exploding multvalued columns.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> pdf = pd.DataFrame({'A': ['1;2', '3', '3;4;5'], 'B':[0] * 3})
        >>> DataFrame(pdf).extract_terms(column='A', sep=';')
           A
        0  1
        1  2
        2  3
        3  4
        4  5

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'xxx, zzz', 'yyy', 'xxx, yyy, zzz']})
        >>> pdf
                 Authors
        0            xxx
        1       xxx, zzz
        2            yyy
        3  xxx, yyy, zzz
        
        >>> DataFrame(pdf).extract_terms(column='Authors')
          Authors
        0     xxx
        1     yyy
        2     zzz

        """
        result = self.explode(column, sep)
        result[column] = result[column].map(lambda x: x.strip())
        result = pd.unique(result[column].dropna())
        result = np.sort(result)
        return pd.DataFrame({column: result})

    def count_terms(self, column, sep=None):
        """Counts the number of different terms in a column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'xxx, zzz', 'yyy', 'xxx, yyy, zzz']})
        >>> pdf
                 Authors
        0            xxx
        1       xxx, zzz
        2            yyy
        3  xxx, yyy, zzz
        
        >>> DataFrame(pdf).count_terms(column='Authors')
        3

        """
        return len(self.extract_terms(column, sep))

    def count_report(self):
        """
        Reports the number of different items per column in dataframe.

        Returns:
            DataFrame.        

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          'Authors':  'xxx,xxx; zzz,yyy; xxx; yyy; zzz'.split(';'),
        ...          'Author(s) ID': '0;1,    3;4;,  4;,  5;,  6;'.split(','),
        ...          'Source title': ' s0,     s0,   s1,  s1, s2'.split(','),
        ...          'Author Keywords': 'k0;k1, k0;k2, k3;k2;k1, k4, k5'.split(','),
        ...          'Index Keywords': 'w0;w1, w0;w2, w3;k1;w1, w4, w5'.split(','),
        ...     }
        ... )
        >>> df
            Authors Author(s) ID Source title Author Keywords Index Keywords
        0   xxx,xxx          0;1           s0           k0;k1          w0;w1
        1   zzz,yyy         3;4;           s0           k0;k2          w0;w2
        2       xxx           4;           s1        k3;k2;k1       w3;k1;w1
        3       yyy           5;           s1              k4             w4
        4       zzz           6;           s2              k5             w5
        
        >>> DataFrame(df).count_report()
                    Column  Number of items
        0          Authors                3
        1     Author(s) ID                7
        2     Source title                3
        3  Author Keywords                6
        4   Index Keywords                7

        """
        columns = [
            "Authors",
            "Author(s) ID",
            "Source title",
            "Author Keywords",
            "Index Keywords",
        ]
        return pd.DataFrame(
            {
                "Column": columns,
                "Number of items": [self.count_terms(w) for w in columns],
            }
        )

    #
    #
    #  Analysis by term
    #
    #

    def summarize_by_term(self, column, sep=None):
        """Summarize the number of documents and citations by term in a dataframe.
        
        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.
        
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3".split(";"),
        ...          "Cited by": list(range(10,14)),
        ...          "ID": list(range(4)),
        ...     }
        ... )
        >>> df
                              Authors  Cited by  ID
        0  author 0,author 1,author 2        10   0
        1                    author 0        11   1
        2                    author 1        12   2
        3                    author 3        13   3

        >>> DataFrame(df).summarize_by_term('Authors', sep=',')
            Authors  Num Documents  Cited by      ID
        0  author 0              2        21  [0, 1]
        1  author 1              2        22  [0, 2]
        2  author 2              1        10     [0]
        3  author 3              1        13     [3]

        """
        data = DataFrame(self[[column, "Cited by", "ID"]]).explode(column, sep)
        data["Num Documents"] = 1
        result = data.groupby(column, as_index=False).agg(
            {"Num Documents": np.size, "Cited by": np.sum}
        )
        result = result.assign(
            ID=data.groupby(column).agg({"ID": list}).reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        result.sort_values(
            [column, "Num Documents", "Cited by"],
            ascending=[True, False, False],
            inplace=True,
            ignore_index=True,
        )
        return result

    def documents_by_term(self, column, sep=None):
        """Computes the number of documents per term in a given column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3".split(";"),
        ...          "Cited by": list(range(10,14)),
        ...          "ID": list(range(4)),
        ...     }
        ... )
        >>> df
                              Authors  Cited by  ID
        0  author 0,author 1,author 2        10   0
        1                    author 0        11   1
        2                    author 1        12   2
        3                    author 3        13   3

        >>> DataFrame(df).documents_by_term('Authors')
            Authors  Num Documents      ID
        0  author 0              2  [0, 1]
        1  author 1              2  [0, 2]
        2  author 2              1     [0]
        3  author 3              1     [3]

        """

        result = self.summarize_by_term(column, sep)
        result.pop("Cited by")
        result.sort_values(
            ["Num Documents", column],
            ascending=[False, True],
            inplace=True,
            ignore_index=True,
        )
        return result

    def citations_by_term(self, column, sep=None):
        """Computes the number of citations by item in a column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3".split(";"),
        ...          "Cited by": list(range(10,14)),
        ...          "ID": list(range(4)),
        ...     }
        ... )
        >>> df
                              Authors  Cited by  ID
        0  author 0,author 1,author 2        10   0
        1                    author 0        11   1
        2                    author 1        12   2
        3                    author 3        13   3

        >>> DataFrame(df).citations_by_term('Authors')
            Authors  Cited by      ID
        0  author 1        22  [0, 2]
        1  author 0        21  [0, 1]
        2  author 3        13     [3]
        3  author 2        10     [0]


        """
        result = self.summarize_by_term(column, sep)
        result.pop("Num Documents")
        result.sort_values(
            ["Cited by", column],
            ascending=[False, True],
            inplace=True,
            ignore_index=True,
        )
        return result

    #
    #
    # Documents and citations by year
    #
    #

    def summarize_by_year(self, cumulative=False):
        """Computes the number of document and the number of total citations per year.
        This funciton adds the missing years in the sequence.

        
        Args:
            cumulative (bool): cumulate values per year.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2016],
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year  Cited by  ID
        0  2010        10   0
        1  2010        11   1
        2  2011        12   2
        3  2011        13   3
        4  2012        14   4
        5  2016        15   5

        >>> DataFrame(df).summarize_by_year()
           Year  Cited by  Num Documents      ID
        0  2010        21              2  [0, 1]
        1  2011        25              2  [2, 3]
        2  2012        14              1     [4]
        3  2013         0              0      []
        4  2014         0              0      []
        5  2015         0              0      []
        6  2016        15              1     [5]

        >>> DataFrame(df).summarize_by_year(cumulative=True)
           Year  Cited by  Num Documents      ID
        0  2010        21              2  [0, 1]
        1  2011        46              4  [2, 3]
        2  2012        60              5     [4]
        3  2013        60              5      []
        4  2014        60              5      []
        5  2015        60              5      []
        6  2016        75              6     [5]


        Comparison with `summarize_by_term`.

        >>> DataFrame(df).summarize_by_term('Year')
           Year  Num Documents  Cited by      ID
        0  2010              2        21  [0, 1]
        1  2011              2        25  [2, 3]
        2  2012              1        14     [4]
        3  2016              1        15     [5]

        """
        data = DataFrame(self[["Year", "Cited by", "ID"]]).explode("Year", None)
        data["Num Documents"] = 1
        result = data.groupby("Year", as_index=False).agg(
            {"Cited by": np.sum, "Num Documents": np.size}
        )
        result = result.assign(
            ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
        result = result.set_index("Year")
        result = result.reindex(years, fill_value=0)
        result["ID"] = result["ID"].map(lambda x: [] if x == 0 else x)
        result.sort_values(
            "Year", ascending=True, inplace=True,
        )
        if cumulative is True:
            result["Num Documents"] = result["Num Documents"].cumsum()
            result["Cited by"] = result["Cited by"].cumsum()
        result = result.reset_index()
        return result

    def documents_by_year(self, cumulative=False):
        """Computes the number of documents per year. This function adds the missing years in the sequence.

        Args:
            cumulative (bool): cumulate values per year.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year  Cited by  ID
        0  2010        10   0
        1  2010        11   1
        2  2011        12   2
        3  2011        13   3
        4  2012        14   4
        5  2014        15   5

        >>> DataFrame(df).documents_by_year()
           Year  Num Documents      ID
        0  2010              2  [0, 1]
        1  2011              2  [2, 3]
        2  2012              1     [4]
        3  2013              0      []
        4  2014              1     [5]

        >>> DataFrame(df).documents_by_year(cumulative=True)
           Year  Num Documents      ID
        0  2010              2  [0, 1]
        1  2011              4  [2, 3]
        2  2012              5     [4]
        3  2013              5      []
        4  2014              6     [5]

        """
        result = self.summarize_by_year(cumulative)
        result.pop("Cited by")
        result = result.reset_index(drop=True)
        return result

    def citations_by_year(self, cumulative=False):
        """Computes the number of citations by year. 
        This function adds the missing years in the sequence.

        Args:
            cumulative (bool): cumulate values per year.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year  Cited by  ID
        0  2010        10   0
        1  2010        11   1
        2  2011        12   2
        3  2011        13   3
        4  2012        14   4
        5  2014        15   5

        >>> DataFrame(df).citations_by_year()
           Year  Cited by      ID
        0  2010        21  [0, 1]
        1  2011        25  [2, 3]
        2  2012        14     [4]
        3  2013         0      []
        4  2014        15     [5]

        >>> DataFrame(df).citations_by_year(cumulative=True)
           Year  Cited by      ID
        0  2010        21  [0, 1]
        1  2011        46  [2, 3]
        2  2012        60     [4]
        3  2013        60      []
        4  2014        75     [5]

        """
        result = self.summarize_by_year(cumulative)
        result.pop("Num Documents")
        result = result.reset_index(drop=True)
        return result

    #
    #
    #  Documents and citations by term per year
    #
    #

    def summarize_by_term_per_year(self, column, sep=None):
        """Computes the number of documents and citations by term per year.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 4;author 4".split(";"),
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year                     Authors  Cited by  ID
        0  2010  author 0,author 1,author 2        10   0
        1  2010                    author 0        11   1
        2  2011                    author 1        12   2
        3  2011                    author 3        13   3
        4  2012                    author 4        14   4
        5  2014                    author 4        15   5

        >>> DataFrame(df).summarize_by_term_per_year('Authors')
            Authors  Year  Cited by  Num Documents      ID
        0  author 0  2010        21              2  [0, 1]
        1  author 1  2010        10              1     [0]
        2  author 2  2010        10              1     [0]
        3  author 1  2011        12              1     [2]
        4  author 3  2011        13              1     [3]
        5  author 4  2012        14              1     [4]
        6  author 4  2014        15              1     [5]

        """
        data = DataFrame(self[["Year", column, "Cited by", "ID"]]).explode(column, sep)
        data["Num Documents"] = 1
        result = data.groupby([column, "Year"], as_index=False).agg(
            {"Cited by": np.sum, "Num Documents": np.size}
        )
        result = result.assign(
            ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        result.sort_values(
            ["Year", column], ascending=True, inplace=True, ignore_index=True,
        )
        return result

    def documents_by_term_per_year(
        self, column, sep=None, as_matrix=False, minmax=None
    ):
        """Computes the number of documents by term per year.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.


        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 4;author 4".split(";"),
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year                     Authors  Cited by  ID
        0  2010  author 0,author 1,author 2        10   0
        1  2010                    author 0        11   1
        2  2011                    author 1        12   2
        3  2011                    author 3        13   3
        4  2012                    author 4        14   4
        5  2014                    author 4        15   5

        >>> DataFrame(df).documents_by_term_per_year('Authors')
            Authors  Year  Num Documents      ID
        0  author 0  2010              2  [0, 1]
        1  author 1  2010              1     [0]
        2  author 2  2010              1     [0]
        3  author 1  2011              1     [2]
        4  author 3  2011              1     [3]
        5  author 4  2012              1     [4]
        6  author 4  2014              1     [5]

        >>> DataFrame(df).documents_by_term_per_year('Authors', as_matrix=True)
              author 0  author 1  author 2  author 3  author 4
        2010         2         1         1         0         0
        2011         0         1         0         1         0
        2012         0         0         0         0         1
        2014         0         0         0         0         1

        >>> DataFrame(df).documents_by_term_per_year('Authors', as_matrix=True, minmax=(2, None))
              author 0
        2010         2

        >>> DataFrame(df).documents_by_term_per_year('Authors', as_matrix=True, minmax=(0, 1))
              author 1  author 2  author 3  author 4
        2010         1         1         0         0
        2011         1         0         1         0
        2012         0         0         0         1
        2014         0         0         0         1

        """

        result = self.summarize_by_term_per_year(column, sep)
        result.pop("Cited by")
        if minmax is not None:
            min_value, max_value = minmax
            if min_value is not None:
                result = result[result["Num Documents"] >= min_value]
            if max_value is not None:
                result = result[result["Num Documents"] <= max_value]
        result.sort_values(
            ["Year", "Num Documents", column],
            ascending=[True, False, True],
            inplace=True,
        )
        result.reset_index(drop=True)
        if as_matrix == True:
            result = pd.pivot_table(
                result,
                values="Num Documents",
                index="Year",
                columns=column,
                fill_value=0,
            )
            result.columns = result.columns.tolist()
            result.index = result.index.tolist()
        return result

    def gant(self, column, sep=None, minmax=None):
        """Computes the number of documents by term per year.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.


        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2011, 2011, 2012, 2015, 2012, 2016],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 3;author 4;author 4".split(";"),
        ...          "Cited by": list(range(10,17)),
        ...          "ID": list(range(7)),
        ...     }
        ... )
        >>> DataFrame(df).documents_by_term_per_year('Authors', as_matrix=True)
              author 0  author 1  author 2  author 3  author 4
        2010         1         1         1         0         0
        2011         1         1         0         0         0
        2012         0         0         0         1         1
        2015         0         0         0         1         0
        2016         0         0         0         0         1

        >>> DataFrame(df).gant('Authors')
              author 0  author 1  author 2  author 3  author 4
        2010         1         1         1         0         0
        2011         1         1         0         0         0
        2012         0         0         0         1         1
        2013         0         0         0         1         1
        2014         0         0         0         1         1
        2015         0         0         0         1         1
        2016         0         0         0         0         1

        """
        result = self.documents_by_term_per_year(
            column=column, sep=sep, as_matrix=True, minmax=minmax
        )
        years = [year for year in range(result.index.min(), result.index.max() + 1)]
        result = result.reindex(years, fill_value=0)

        matrix1 = result.copy()
        matrix1 = matrix1.cumsum()
        matrix1 = matrix1.applymap(lambda x: True if x > 0 else False)

        matrix2 = result.copy()
        matrix2 = matrix2.sort_index(ascending=False)
        matrix2 = matrix2.cumsum()
        matrix2 = matrix2.applymap(lambda x: True if x > 0 else False)
        matrix2 = matrix2.sort_index(ascending=True)
        result = matrix1.eq(matrix2)
        result = result.applymap(lambda x: 1 if x is True else 0)

        return result

    def citations_by_term_per_year(
        self, column, sep=None, as_matrix=False, minmax=None
    ):
        """Computes the number of citations by term by year in a column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.


        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 4;author 4".split(";"),
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year                     Authors  Cited by  ID
        0  2010  author 0,author 1,author 2        10   0
        1  2010                    author 0        11   1
        2  2011                    author 1        12   2
        3  2011                    author 3        13   3
        4  2012                    author 4        14   4
        5  2014                    author 4        15   5

        >>> DataFrame(df).citations_by_term_per_year('Authors')
            Authors  Year  Cited by      ID
        0  author 0  2010        21  [0, 1]
        1  author 2  2010        10     [0]
        2  author 1  2010        10     [0]
        3  author 3  2011        13     [3]
        4  author 1  2011        12     [2]
        5  author 4  2012        14     [4]
        6  author 4  2014        15     [5]

        >>> DataFrame(df).citations_by_term_per_year('Authors', as_matrix=True)
              author 0  author 1  author 2  author 3  author 4
        2010        21        10        10         0         0
        2011         0        12         0        13         0
        2012         0         0         0         0        14
        2014         0         0         0         0        15

        >>> DataFrame(df).citations_by_term_per_year('Authors', as_matrix=True, minmax=(12, 15))
              author 1  author 3  author 4
        2011        12        13         0
        2012         0         0        14
        2014         0         0        15


        """
        result = self.summarize_by_term_per_year(column, sep)
        result.pop("Num Documents")
        if minmax is not None:
            min_value, max_value = minmax
            if min_value is not None:
                result = result[result["Cited by"] >= min_value]
            if max_value is not None:
                result = result[result["Cited by"] <= max_value]
        result.sort_values(
            ["Year", "Cited by", column], ascending=[True, False, False], inplace=True,
        )
        result = result.reset_index(drop=True)
        if as_matrix == True:
            result = pd.pivot_table(
                result, values="Cited by", index="Year", columns=column, fill_value=0,
            )
            result.columns = result.columns.tolist()
            result.index = result.index.tolist()
        return result

    #
    #
    #  Documents and citations by term per term per year
    #
    #

    def summarize_by_term_per_term_per_year(
        self, column_IDX, column_COL, sep_IDX=None, sep_COL=None
    ):
        """Computes the number of documents and citations by term per term by year.

        Args:
            column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 4;author 4".split(";"),
        ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year                     Authors Author Keywords  Cited by  ID
        0  2010  author 0,author 1,author 2           w0;w1        10   0
        1  2010                    author 0              w0        11   1
        2  2011                    author 1              w1        12   2
        3  2011                    author 3        w5;w3;w4        13   3
        4  2012                    author 4              w5        14   4
        5  2014                    author 4              w3        15   5

        >>> DataFrame(df).summarize_by_term_per_term_per_year('Authors', 'Author Keywords')
             Authors Author Keywords  Year  Cited by  Num Documents      ID
        0   author 0              w0  2010        21              2  [0, 1]
        1   author 0              w1  2010        10              1     [0]
        2   author 1              w0  2010        10              1     [0]
        3   author 1              w1  2010        10              1     [0]
        4   author 2              w0  2010        10              1     [0]
        5   author 2              w1  2010        10              1     [0]
        6   author 1              w1  2011        12              1     [2]
        7   author 3              w3  2011        13              1     [3]
        8   author 3              w4  2011        13              1     [3]
        9   author 3              w5  2011        13              1     [3]
        10  author 4              w5  2012        14              1     [4]
        11  author 4              w3  2014        15              1     [5]

        """

        data = DataFrame(
            self[[column_IDX, column_COL, "Year", "Cited by", "ID"]]
        ).explode(column_IDX, sep_IDX)
        data = DataFrame(data).explode(column_COL, sep_COL)
        data["Num Documents"] = 1
        result = data.groupby([column_IDX, column_COL, "Year"], as_index=False).agg(
            {"Cited by": np.sum, "Num Documents": np.size}
        )
        result = result.assign(
            ID=data.groupby([column_IDX, column_COL, "Year"])
            .agg({"ID": list})
            .reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        result.sort_values(
            ["Year", column_IDX, column_COL,], ascending=True, inplace=True
        )
        return result.reset_index(drop=True)

    def documents_by_terms_per_terms_per_year(
        self, column_IDX, column_COL, sep_IDX=None, sep_COL=None
    ):
        """Computes the number of documents by term per term per year.

        Args:
            column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 4;author 4".split(";"),
        ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year                     Authors Author Keywords  Cited by  ID
        0  2010  author 0,author 1,author 2           w0;w1        10   0
        1  2010                    author 0              w0        11   1
        2  2011                    author 1              w1        12   2
        3  2011                    author 3        w5;w3;w4        13   3
        4  2012                    author 4              w5        14   4
        5  2014                    author 4              w3        15   5

        >>> DataFrame(df).documents_by_terms_per_terms_per_year('Authors', 'Author Keywords')
             Authors Author Keywords  Year  Num Documents      ID
        0   author 0              w0  2010              2  [0, 1]
        1   author 0              w1  2010              1     [0]
        2   author 1              w0  2010              1     [0]
        3   author 1              w1  2010              1     [0]
        4   author 2              w0  2010              1     [0]
        5   author 2              w1  2010              1     [0]
        6   author 1              w1  2011              1     [2]
        7   author 3              w3  2011              1     [3]
        8   author 3              w4  2011              1     [3]
        9   author 3              w5  2011              1     [3]
        10  author 4              w5  2012              1     [4]
        11  author 4              w3  2014              1     [5]


        """

        result = self.summarize_by_term_per_term_per_year(
            column_IDX, column_COL, sep_IDX, sep_COL
        )
        result.pop("Cited by")
        result.sort_values(
            ["Year", column_IDX, column_COL],
            ascending=[True, True, True],
            inplace=True,
        )
        return result.reset_index(drop=True)

    def citations_by_terms_per_terms_per_year(
        self, column_IDX, column_COL, sep_IDX=None, sep_COL=None
    ):
        """Computes the number of citations by term per term per year.

        Args:
            column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
        ...          "Authors": "author 0,author 1,author 2;author 0;author 1;author 3;author 4;author 4".split(";"),
        ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
        ...          "Cited by": list(range(10,16)),
        ...          "ID": list(range(6)),
        ...     }
        ... )
        >>> df
           Year                     Authors Author Keywords  Cited by  ID
        0  2010  author 0,author 1,author 2           w0;w1        10   0
        1  2010                    author 0              w0        11   1
        2  2011                    author 1              w1        12   2
        3  2011                    author 3        w5;w3;w4        13   3
        4  2012                    author 4              w5        14   4
        5  2014                    author 4              w3        15   5

        >>> DataFrame(df).citations_by_terms_per_terms_per_year('Authors', 'Author Keywords')
             Authors Author Keywords  Year  Cited by      ID
        0   author 0              w0  2010        21  [0, 1]
        1   author 0              w1  2010        10     [0]
        2   author 1              w0  2010        10     [0]
        3   author 1              w1  2010        10     [0]
        4   author 2              w0  2010        10     [0]
        5   author 2              w1  2010        10     [0]
        6   author 1              w1  2011        12     [2]
        7   author 3              w3  2011        13     [3]
        8   author 3              w4  2011        13     [3]
        9   author 3              w5  2011        13     [3]
        10  author 4              w5  2012        14     [4]
        11  author 4              w3  2014        15     [5]


        """

        result = self.summarize_by_term_per_term_per_year(
            column_IDX, column_COL, sep_IDX, sep_COL
        )
        result.pop("Num Documents")
        result.sort_values(
            ["Year", column_IDX, column_COL],
            ascending=[True, True, True],
            inplace=True,
        )
        return result.reset_index(drop=True)

    #
    #
    #  Co-occurrence
    #
    #

    def summarize_co_occurrence(
        self, column_IDX, column_COL, sep_IDX=None, sep_COL=None
    ):
        """Summarize occurrence and citations by terms in two different columns.

        Args:
            column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4

        >>> DataFrame(df).summarize_co_occurrence(column_IDX='Authors', column_COL='Author Keywords')
          Authors (IDX) Author Keywords (COL)  Num Documents  Cited by      ID
        0             A                     a              2         1  [0, 1]
        1             A                     b              1         1     [1]
        2             A                     c              1         3     [3]
        3             B                     a              1         1     [1]
        4             B                     b              2         3  [1, 2]
        5             B                     c              2         7  [3, 4]
        6             B                     d              1         4     [4]
        7             C                     c              1         3     [3]
        8             D                     c              1         4     [4]
        9             D                     d              1         4     [4]

        """

        def generate_pairs(w, v):
            if sep_IDX is not None:
                w = [x.strip() for x in w.split(sep_IDX)]
            else:
                w = [w]
            if sep_COL is not None:
                v = [x.strip() for x in v.split(sep_COL)]
            else:
                v = [v]
            result = []
            for idx0 in range(len(w)):
                for idx1 in range(len(v)):
                    result.append((w[idx0], v[idx1]))
            return result

        if sep_IDX is None and column_IDX in SCOPUS_SEPS:
            sep_IDX = SCOPUS_SEPS[column_IDX]

        if sep_COL is None and column_COL in SCOPUS_SEPS:
            sep_COL = SCOPUS_SEPS[column_COL]

        data = self.copy()
        data = data[[column_IDX, column_COL, "Cited by", "ID"]]
        data["Num Documents"] = 1
        data["pairs"] = [
            generate_pairs(a, b) for a, b in zip(data[column_IDX], data[column_COL])
        ]
        data = data[["pairs", "Num Documents", "Cited by", "ID"]]
        data = data.explode("pairs")

        result = data.groupby("pairs", as_index=False).agg(
            {"Cited by": np.sum, "Num Documents": np.sum, "ID": list}
        )

        result["Cited by"] = result["Cited by"].map(int)

        result[column_IDX + " (IDX)"] = result["pairs"].map(lambda x: x[0])
        result[column_COL + " (COL)"] = result["pairs"].map(lambda x: x[1])
        result.pop("pairs")

        result = result[
            [
                column_IDX + " (IDX)",
                column_COL + " (COL)",
                "Num Documents",
                "Cited by",
                "ID",
            ]
        ]

        result = result.sort_values(
            [column_IDX + " (IDX)", column_COL + " (COL)"], ignore_index=True,
        )

        return result

    def co_occurrence(
        self,
        column_IDX,
        column_COL,
        sep_IDX=None,
        sep_COL=None,
        as_matrix=False,
        minmax=None,
    ):
        """Computes the co-occurrence of two terms in different colums. The report adds
        the number of documents by term between brackets.

        Args:
            column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.
            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4

        >>> DataFrame(df).co_occurrence(column_IDX='Authors', column_COL='Author Keywords')
          Authors (IDX) Author Keywords (COL)  Num Documents      ID
        0             A                     a              2  [0, 1]
        1             B                     b              2  [1, 2]
        2             B                     c              2  [3, 4]
        3             A                     b              1     [1]
        4             A                     c              1     [3]
        5             B                     a              1     [1]
        6             B                     d              1     [4]
        7             C                     c              1     [3]
        8             D                     c              1     [4]
        9             D                     d              1     [4]

        >>> DataFrame(df).co_occurrence(column_IDX='Authors', column_COL='Author Keywords', as_matrix=True)
           a  b  c  d
        A  2  1  1  0
        B  1  2  2  1
        C  0  0  1  0
        D  0  0  1  1

        >>> DataFrame(df).co_occurrence(column_IDX='Authors', column_COL='Author Keywords', as_matrix=True, minmax=(2,2))
           a  b  c
        A  2  0  0
        B  0  2  2

        """

        def generate_dic(column, sep):
            new_names = self.documents_by_term(column, sep)
            new_names = {
                term: "{:s} [{:d}]".format(term, docs_per_term)
                for term, docs_per_term in zip(
                    new_names[column], new_names["Num Documents"],
                )
            }
            return new_names

        result = self.summarize_co_occurrence(column_IDX, column_COL, sep_IDX, sep_COL)
        result.pop("Cited by")
        if minmax is not None:
            min_value, max_value = minmax
            if min_value is not None:
                result = result[result["Num Documents"] >= min_value]
            if max_value is not None:
                result = result[result["Num Documents"] <= max_value]
        result.sort_values(
            [column_IDX + " (IDX)", column_COL + " (COL)", "Num Documents",],
            ascending=[True, True, False],
            inplace=True,
        )
        result = result.sort_values(
            ["Num Documents", column_IDX + " (IDX)", column_COL + " (COL)"],
            ascending=[False, True, True],
        )
        result = result.reset_index(drop=True)
        if as_matrix == True:
            result = pd.pivot_table(
                result,
                values="Num Documents",
                index=column_IDX + " (IDX)",
                columns=column_COL + " (COL)",
                fill_value=0,
            )
            result.columns = result.columns.tolist()
            result.index = result.index.tolist()
        return result

    def co_citation(
        self,
        column_IDX,
        column_COL,
        sep_IDX=None,
        sep_COL=None,
        as_matrix=False,
        minmax=None,
    ):
        """Computes the number of citations shared by two terms in different columns.

        Args:
            column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.
            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4

        >>> DataFrame(df).co_citation(column_IDX='Authors', column_COL='Author Keywords')
          Authors (IDX) Author Keywords (COL)  Cited by      ID
        0             B                     c         7  [3, 4]
        1             B                     d         4     [4]
        2             D                     c         4     [4]
        3             D                     d         4     [4]
        4             A                     c         3     [3]
        5             B                     b         3  [1, 2]
        6             C                     c         3     [3]
        7             A                     a         1  [0, 1]
        8             A                     b         1     [1]
        9             B                     a         1     [1]

        >>> DataFrame(df).co_citation(column_IDX='Authors', column_COL='Author Keywords', as_matrix=True)
           a  b  c  d
        A  1  1  3  0
        B  1  3  7  4
        C  0  0  3  0
        D  0  0  4  4

        >>> DataFrame(df).co_citation(column_IDX='Authors', column_COL='Author Keywords', as_matrix=True, minmax=(3,4))
           b  c  d
        A  0  3  0
        B  3  0  4
        C  0  3  0
        D  0  4  4

        """

        def generate_dic(column, sep):
            new_names = self.citations_by_term(column, sep)
            new_names = {
                term: "{:s} [{:d}]".format(term, docs_per_term)
                for term, docs_per_term in zip(
                    new_names[column], new_names["Cited by"],
                )
            }
            return new_names

        result = self.summarize_co_occurrence(column_IDX, column_COL, sep_IDX, sep_COL)
        result.pop("Num Documents")
        if minmax is not None:
            min_value, max_value = minmax
            if min_value is not None:
                result = result[result["Cited by"] >= min_value]
            if max_value is not None:
                result = result[result["Cited by"] <= max_value]
        result.sort_values(
            ["Cited by", column_IDX + " (IDX)", column_COL + " (COL)",],
            ascending=[False, True, True,],
            inplace=True,
        )
        result = result.reset_index(drop=True)
        if as_matrix == True:
            result = pd.pivot_table(
                result,
                values="Cited by",
                index=column_IDX + " (IDX)",
                columns=column_COL + " (COL)",
                fill_value=0,
            )
            result.columns = result.columns.tolist()
            result.index = result.index.tolist()
        return result

    #
    #
    #  Occurrence
    #
    #

    def summarize_occurrence(self, column, sep=None):
        """Summarize occurrence and citations by terms in a column of a dataframe.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A', 'A,B', 'B', 'A,B,C', 'D', 'B,D']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors  Cited by  ID
        0       A         0   0
        1       A         1   1
        2     A,B         2   2
        3       B         3   3
        4   A,B,C         4   4
        5       D         5   5
        6     B,D         6   6

        >>> DataFrame(df).summarize_occurrence(column='Authors')
           Authors (IDX) Authors (COL)  Num Documents  Cited by            ID
        0              A             A              4         7  [0, 1, 2, 4]
        1              A             B              2         6        [2, 4]
        2              A             C              1         4           [4]
        3              B             A              2         6        [2, 4]
        4              B             B              4        15  [2, 3, 4, 6]
        5              B             C              1         4           [4]
        6              B             D              1         6           [6]
        7              C             A              1         4           [4]
        8              C             B              1         4           [4]
        9              C             C              1         4           [4]
        10             D             B              1         6           [6]
        11             D             D              2        11        [5, 6]
        
        """

        def generate_pairs(w):
            w = [x.strip() for x in w.split(sep)]
            result = []
            for idx0 in range(len(w)):
                # for idx1 in range(idx0, len(w)):
                for idx1 in range(len(w)):
                    result.append((w[idx0], w[idx1]))
            return result

        if sep is None and column in SCOPUS_SEPS:
            sep = SCOPUS_SEPS[column]

        data = self.copy()
        data = data[[column, "Cited by", "ID"]]
        data["count"] = 1

        data["pairs"] = data[column].map(lambda x: generate_pairs(x))
        data = data[["pairs", "count", "Cited by", "ID"]]
        data = data.explode("pairs")

        result = (
            data.groupby("pairs", as_index=False)
            .agg({"Cited by": np.sum, "count": np.sum, "ID": list})
            .rename(columns={"count": "Num Documents"})
        )

        result["Cited by"] = result["Cited by"].map(int)

        result[column + " (IDX)"] = result["pairs"].map(lambda x: x[0])
        result[column + " (COL)"] = result["pairs"].map(lambda x: x[1])
        result.pop("pairs")

        result = result[
            [column + " (IDX)", column + " (COL)", "Num Documents", "Cited by", "ID"]
        ]

        result = result.sort_values(
            [column + " (IDX)", column + " (COL)"], ignore_index=True,
        )

        return result

    def occurrence(self, column, sep=None, as_matrix=False, minmax=None):
        """Computes the occurrence between the terms in a column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A', 'A,B', 'B', 'A,B,C', 'D', 'B,D']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors  Cited by  ID
        0       A         0   0
        1       A         1   1
        2     A,B         2   2
        3       B         3   3
        4   A,B,C         4   4
        5       D         5   5
        6     B,D         6   6

        >>> DataFrame(df).occurrence(column='Authors')
           Authors (IDX) Authors (COL)  Num Documents            ID
        0              A             A              4  [0, 1, 2, 4]
        1              B             B              4  [2, 3, 4, 6]
        2              A             B              2        [2, 4]
        3              B             A              2        [2, 4]
        4              D             D              2        [5, 6]
        5              A             C              1           [4]
        6              B             C              1           [4]
        7              B             D              1           [6]
        8              C             A              1           [4]
        9              C             B              1           [4]
        10             C             C              1           [4]
        11             D             B              1           [6]

        >>> DataFrame(df).occurrence(column='Authors', as_matrix=True)
           A  B  C  D
        A  4  2  1  0
        B  2  4  1  1
        C  1  1  1  0
        D  0  1  0  2

        >>> DataFrame(df).occurrence(column='Authors', as_matrix=True, minmax=(2,3))
           A  B  D
        A  0  2  0
        B  2  0  0
        D  0  0  2

        """

        def generate_dic(column, sep):
            new_names = self.documents_by_term(column, sep)
            new_names = {
                term: "{:s} [{:d}]".format(term, docs_per_term)
                for term, docs_per_term in zip(
                    new_names[column], new_names["Num Documents"],
                )
            }
            return new_names

        column_IDX = column + " (IDX)"
        column_COL = column + " (COL)"

        result = self.summarize_occurrence(column, sep)
        result.pop("Cited by")
        if minmax is not None:
            min_value, max_value = minmax
            if min_value is not None:
                result = result[result["Num Documents"] >= min_value]
            if max_value is not None:
                result = result[result["Num Documents"] <= max_value]
        result.sort_values(
            ["Num Documents", column_IDX, column_COL],
            ascending=[False, True, True],
            inplace=True,
        )
        result = result.reset_index(drop=True)
        if as_matrix == True:
            result = pd.pivot_table(
                result,
                values="Num Documents",
                index=column_IDX,
                columns=column_COL,
                fill_value=0,
            )
            result.columns = result.columns.tolist()
            result.index = result.index.tolist()
        return result

    def compute_occurrence_map(self, column, sep=None, minmax=None):
        """Computes a occurrence between terms in a column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            dictionary
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A', 'A,B', 'B', 'A,B,C', 'D', 'B,D']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors  ID
        0       A   0
        1       A   1
        2     A,B   2
        3       B   3
        4   A,B,C   4
        5       D   5
        6     B,D   6

        >>> DataFrame(df).compute_occurrence_map(column='Authors')
        {'terms': ['A', 'B', 'C', 'D'], 'docs': ['doc#0', 'doc#1', 'doc#2', 'doc#3', 'doc#4', 'doc#5'], 'edges': [('A', 'doc#0'), ('A', 'doc#1'), ('B', 'doc#1'), ('A', 'doc#2'), ('B', 'doc#2'), ('C', 'doc#2'), ('B', 'doc#3'), ('B', 'doc#4'), ('D', 'doc#4'), ('D', 'doc#5')], 'labels': {'doc#0': 2, 'doc#1': 1, 'doc#2': 1, 'doc#3': 1, 'doc#4': 1, 'doc#5': 1, 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}}

        """
        result = self[[column]].copy()
        result["count"] = 1

        result = result.groupby(column, as_index=False).agg({"count": np.sum})

        if sep is None and column in SCOPUS_SEPS.keys():
            sep = SCOPUS_SEPS[column]
        if sep is not None:
            result[column] = result[column].map(
                lambda x: sorted(x.split(sep)) if isinstance(x, str) else x
            )

        result["doc-ID"] = ["doc#{:d}".format(i) for i in range(len(result))]

        terms = result[[column]].copy()
        terms.explode(column)
        terms = [item for sublist in terms[column].tolist() for item in sublist]
        terms = sorted(set(terms))
        docs = result["doc-ID"].tolist()
        label_docs = {doc: label for doc, label in zip(docs, result["count"].tolist())}
        label_terms = {t: t for t in terms}
        labels = {**label_docs, **label_terms}

        edges = []
        for field, docID in zip(result[column], result["doc-ID"]):
            for item in field:
                edges.append((item, docID))

        return dict(terms=terms, docs=docs, edges=edges, labels=labels)

    #
    #
    #  Analytical functions
    #
    #

    def compute_tfm(self, column, sep=None):
        """Computes the term-frequency matrix for the terms in a column.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.

        Returns:
            DataFrame.
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4

        >>> DataFrame(df).compute_tfm('Authors')
           A  B  C  D
        0  1  0  0  0
        1  1  1  0  0
        2  0  1  0  0
        3  1  1  1  0
        4  0  1  0  1

        >>> DataFrame(df).compute_tfm('Author Keywords')
           a  b  c  d
        0  1  0  0  0
        1  1  1  0  0
        2  0  1  0  0
        3  0  0  1  0
        4  0  0  1  1
        
        """
        data = self[[column, "ID"]].copy()
        data["value"] = 1.0
        data = DataFrame(data).explode(column, sep)

        result = pd.pivot_table(
            data=data, index="ID", columns=column, margins=False, fill_value=0.0,
        )
        result.columns = [b for _, b in result.columns]

        result = result.reset_index(drop=True)

        return result

    def auto_corr(
        self, column, sep=None, method="pearson", as_matrix=True, minmax=None
    ):
        """Computes the autocorrelation among items in a column of the dataframe.

        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            method (str): Available methods are:

                * pearson : Standard correlation coefficient.

                * kendall : Kendall Tau correlation coefficient.

                * spearman : Spearman rank correlation.

            as_matrix (bool): Results are returned as a matrix.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.
            
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D', 'A,B']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4
        5     A,B               d         5   5

        
        >>> DataFrame(df).auto_corr('Authors')
                  A         B         C         D
        A  1.000000 -0.316228  0.316228 -0.632456
        B -0.316228  1.000000  0.200000  0.200000
        C  0.316228  0.200000  1.000000 -0.200000
        D -0.632456  0.200000 -0.200000  1.000000
        
        >>> DataFrame(df).auto_corr('Authors', as_matrix=False)
           Authors (IDX) Authors (COL)     value
        0              A             A  1.000000
        1              B             A -0.316228
        2              C             A  0.316228
        3              D             A -0.632456
        4              A             B -0.316228
        5              B             B  1.000000
        6              C             B  0.200000
        7              D             B  0.200000
        8              A             C  0.316228
        9              B             C  0.200000
        10             C             C  1.000000
        11             D             C -0.200000
        12             A             D -0.632456
        13             B             D  0.200000
        14             C             D -0.200000
        15             D             D  1.000000

        >>> DataFrame(df).auto_corr('Author Keywords')
              a     b     c     d
        a  1.00  0.25 -0.50 -0.50
        b  0.25  1.00 -0.50 -0.50
        c -0.50 -0.50  1.00  0.25
        d -0.50 -0.50  0.25  1.00

        >>> DataFrame(df).auto_corr('Author Keywords', minmax=(0.25, None))
             a    b     c     d
        a  1.0  0.0  0.00  0.00
        b  0.0  1.0  0.00  0.00
        c  0.0  0.0  1.00  0.25
        d  0.0  0.0  0.25  1.00

        """
        result = self.compute_tfm(column=column, sep=sep)
        result = result.corr(method=method)
        if as_matrix is False or minmax is not None:
            result = (
                result.reset_index()
                .melt("index")
                .rename(
                    columns={"index": column + " (IDX)", "variable": column + " (COL)"}
                )
            )
            if minmax is not None:
                min_value, max_value = minmax
                if min_value is not None:
                    result = result[result["value"] >= float(min_value)]
                if max_value is not None:
                    result = result[result["value"] <= float(max_value)]
            if as_matrix is True:
                result = pd.pivot_table(
                    result,
                    values="value",
                    index=column + " (IDX)",
                    columns=column + " (COL)",
                    fill_value=float(0.0),
                )
                result.columns = result.columns.tolist()
                result.index = result.index.tolist()
                result = result.astype("float")

        return result

    def cross_corr(
        self,
        column_IDX,
        column_COL=None,
        sep_IDX=None,
        sep_COL=None,
        method="pearson",
        as_matrix=True,
        minmax=None,
    ):
        """Computes the cross-correlation among items in two different columns of the dataframe.

        Args:
            column_IDX (str): the first column.
            sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
            column_COL (str): the second column.
            sep_COL (str): Character used as internal separator for the elements in the column_COL.
            method (str): Available methods are:
                
                - pearson : Standard correlation coefficient.
                
                - kendall : Kendall Tau correlation coefficient.
                
                - spearman : Spearman rank correlation.

            as_matrix (bool): the result is reshaped by melt or not.
            minmax (pair(number,number)): filter values by >=min,<=max.

        Returns:
            DataFrame.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D', 'A,B']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4
        5     A,B               d         5   5


        >>> DataFrame(df).compute_tfm('Authors')
           A  B  C  D
        0  1  0  0  0
        1  1  1  0  0
        2  0  1  0  0
        3  1  1  1  0
        4  0  1  0  1
        5  1  1  0  0

        >>> DataFrame(df).compute_tfm('Author Keywords')
           a  b  c  d
        0  1  0  0  0
        1  1  1  0  0
        2  0  1  0  0
        3  0  0  1  0
        4  0  0  1  1
        5  0  0  0  1


        >>> DataFrame(df).cross_corr('Authors', 'Author Keywords')
              a         b         c         d
        A  0.50 -0.632456 -0.316228 -0.316228
        B -0.25  0.316228 -0.316228 -0.316228
        C -0.25  0.316228  0.632456  0.632456
        D -0.25  0.316228 -0.316228  0.632456

        >>> DataFrame(df).cross_corr('Authors', 'Author Keywords', minmax=(0.45, 0.8))
             a         c         d
        A  0.5  0.000000  0.000000
        C  0.0  0.632456  0.632456
        D  0.0  0.000000  0.632456

        >>> DataFrame(df).cross_corr('Authors', 'Author Keywords', as_matrix=False)
           Authors Author Keywords     value
        0        A               a  0.500000
        1        B               a -0.250000
        2        C               a -0.250000
        3        D               a -0.250000
        4        A               b -0.632456
        5        B               b  0.316228
        6        C               b  0.316228
        7        D               b  0.316228
        8        A               c -0.316228
        9        B               c -0.316228
        10       C               c  0.632456
        11       D               c -0.316228
        12       A               d -0.316228
        13       B               d -0.316228
        14       C               d  0.632456
        15       D               d  0.632456

        """
        if column_IDX == column_COL:
            return self.auto_corr(
                column=column_IDX,
                sep=sep_IDX,
                method=method,
                as_matrix=as_matrix,
                minmax=minmax,
            )
        tfm_r = self.compute_tfm(column=column_IDX, sep=sep_IDX)
        tfm_c = self.compute_tfm(column=column_COL, sep=sep_COL)
        result = pd.DataFrame(
            [
                [tfm_r[row].corr(tfm_c[col]) for row in tfm_r.columns]
                for col in tfm_c.columns
            ],
            columns=tfm_c.columns,
            index=tfm_r.columns,
        )

        if as_matrix is False or minmax is not None:
            result = (
                result.reset_index()
                .melt("index")
                .rename(columns={"index": column_IDX, "variable": column_COL})
            )
            if minmax is not None:
                min_value, max_value = minmax
                if min_value is not None:
                    result = result[result["value"] >= float(min_value)]
                if max_value is not None:
                    result = result[result["value"] <= float(max_value)]
            if as_matrix is True:
                result = pd.pivot_table(
                    result,
                    values="value",
                    index=column_IDX,
                    columns=column_COL,
                    fill_value=float(0.0),
                )
                result.columns = result.columns.tolist()
                result.index = result.index.tolist()
                result = result.astype("float")
        return result

    def factor_analysis(self, column, sep=None, n_components=None, as_matrix=True):
        """Computes the matrix of factors for terms in a given column.


        Args:
            column (str): the column to explode.
            sep (str): Character used as internal separator for the elements in the column.
            n_components: Number of components to compute.
            as_matrix (bool): the result is reshaped by melt or not.

        Returns:
            DataFrame.
       
        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> x = [ 'A', 'A,B', 'B', 'A,B,C', 'B,D', 'A,B']
        >>> y = [ 'a', 'a;b', 'b', 'c', 'c;d', 'd']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'Author Keywords': y,
        ...       'Cited by': list(range(len(x))),
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors Author Keywords  Cited by  ID
        0       A               a         0   0
        1     A,B             a;b         1   1
        2       B               b         2   2
        3   A,B,C               c         3   3
        4     B,D             c;d         4   4
        5     A,B               d         5   5


        >>> DataFrame(df).compute_tfm('Authors')
           A  B  C  D
        0  1  0  0  0
        1  1  1  0  0
        2  0  1  0  0
        3  1  1  1  0
        4  0  1  0  1
        5  1  1  0  0


        >>> DataFrame(df).factor_analysis('Authors', n_components=3)
                 F0            F1       F2
        A -0.774597 -0.000000e+00  0.00000
        B  0.258199  7.071068e-01 -0.57735
        C -0.258199  7.071068e-01  0.57735
        D  0.516398  1.110223e-16  0.57735

        >>> DataFrame(df).factor_analysis('Authors', n_components=3, as_matrix=False)
           Authors Factor         value
        0        A     F0 -7.745967e-01
        1        B     F0  2.581989e-01
        2        C     F0 -2.581989e-01
        3        D     F0  5.163978e-01
        4        A     F1 -0.000000e+00
        5        B     F1  7.071068e-01
        6        C     F1  7.071068e-01
        7        D     F1  1.110223e-16
        8        A     F2  0.000000e+00
        9        B     F2 -5.773503e-01
        10       C     F2  5.773503e-01
        11       D     F2  5.773503e-01

        """

        tfm = self.compute_tfm(column, sep)
        terms = tfm.columns.tolist()
        if n_components is None:
            n_components = int(np.sqrt(len(set(terms))))
        pca = PCA(n_components=n_components)
        result = np.transpose(pca.fit(X=tfm.values).components_)
        result = pd.DataFrame(
            result, columns=["F" + str(i) for i in range(n_components)], index=terms
        )
        if as_matrix is True:
            return result
        return (
            result.reset_index()
            .melt("index")
            .rename(columns={"index": column, "variable": "Factor"})
        )
