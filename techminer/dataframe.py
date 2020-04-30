"""
TechMiner.DataFrame
==================================================================================================




"""
import pandas as pd
import numpy as np

SCOPUS_SEPS = {"Authors": ",", "Author Keywords": ";", "Index Keywords": ";", "ID": ";"}


def _expand(pdf, column, sep):

    result = pdf.copy()
    if sep is None and column in SCOPUS_SEPS.keys():
        sep = SCOPUS_SEPS[column]
    if sep is not None:
        result[column] = result[column].map(
            lambda x: x.split(sep) if isinstance(x, str) else x
        )
        result = result.explode(column)
        result[column] = result[column].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        result.index = list(range(len(result)))

    return result


class DataFrame(pd.DataFrame):
    """Class to represent a dataframe of bibliographic records.
    """

    # ----------------------------------------------------------------------
    # Compatitbility with pandas.DataFrame
    #
    @property
    def _constructor_expanddim(self):
        return self

    #
    # Numeration of documents for referencing
    #
    def generate_ID(self, fmt=None):
        """Generates a unique ID for each document.
        """
        if fmt is None:
            self["ID"] = [x for x in range(len(self))]
        else:
            self["ID"] = [fmt.format(x) for x in range(len(self))]
        self.index = list(range(len(self)))
        return self

    #
    # Basic info
    #

    def coverage(self):
        """Counts the number of None values per column.


        Returns:
            Pandas DataFrame.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data)
        >>> rdf.coverage()
                         Field  Number of items Coverage (%)
        0              Authors              144      100.00%
        1         Author(s) ID              144      100.00%
        2                Title              144      100.00%
        3                 Year              144      100.00%
        4         Source title              144      100.00%
        5               Volume               97       67.36%
        6                Issue               27       18.75%
        7             Art. No.               49       34.03%
        8           Page start              119       82.64%
        9             Page end              119       82.64%
        10          Page count                0        0.00%
        11            Cited by               68       47.22%
        12                 DOI              133       92.36%
        13        Affiliations              143       99.31%
        14       Document Type              144      100.00%
        15         Access Type               16       11.11%
        16              Source              144      100.00%
        17                 EID              144      100.00%
        18            Abstract              144      100.00%
        19     Author Keywords              124       86.11%
        20      Index Keywords              123       85.42%
        21          References              137       95.14%
        22            keywords              144      100.00%
        23                CONF              144      100.00%
        24  keywords (cleaned)              144      100.00%
        25            SELECTED              144      100.00%
        26                  ID              144      100.00%

        """

        return pd.DataFrame(
            {
                "Field": self.columns,
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
    # Term extraction
    #

    def extract_terms(self, column, sep=None):
        """

        >>> import pandas as pd
        >>> pdf = pd.DataFrame({'A': ['1;2', '3', '3;4;5'], 'B':[0] * 3})
        >>> DataFrame(pdf).extract_terms(column='A', sep=';')
           A
        0  1
        1  2
        2  3
        3  4
        4  5

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'yyy', 'xxx, zzz', 'xxx, yyy, zzz']})
        >>> DataFrame(pdf).extract_terms(column='Authors')
          Authors
        0     xxx
        1     yyy
        2     zzz

        """
        result = _expand(self, column, sep)
        result = pd.unique(result[column])
        result = np.sort(result)
        return pd.DataFrame({column: result})

    def count_terms(self, column, sep=None):
        """

        >>> import pandas as pd
        >>> pdf = pd.DataFrame({'A': ['1;2', '3', '3;4;5'], 'B':[0] * 3})
        >>> DataFrame(pdf).count_terms(column='A', sep=';')
        5

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'yyy', 'xxx, zzz', 'xxx, yyy, zzz']})
        >>> DataFrame(pdf).count_terms(column='Authors')
        3

        """
        return len(self.extract_terms(column, sep))

    #
    #
    # Document counting
    #
    #

    def documents_by_term(self, column, sep=None, top_n=None):
        """Computes the number of documents per term in a given column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.documents_by_term('Authors').head(5)
                Authors  Num Documents                             ID
        0       Wang J.              7  [3, 10, 15, 80, 87, 128, 128]
        1      Zhang G.              4             [27, 78, 117, 119]
        2        Yan X.              3                   [13, 44, 85]
        3  Hernandez G.              3                  [52, 94, 100]
        4      Tefas A.              3                  [8, 110, 114]

        """

        data = _expand(self[[column, "ID"]], column, sep)

        result = (
            data.groupby(column, as_index=False)
            .agg({"ID": np.size})
            .rename(columns={"ID": "Num Documents"})
        )

        result = result.assign(
            ID=data.groupby(column).agg({"ID": list}).reset_index()["ID"]
        )
        result.sort_values("Num Documents", ascending=False, inplace=True)

        if top_n is not None:
            result = result.head(top_n)

        result.index = list(range(len(result)))

        return result

    #
    #
    #

    def documents_by_year(self, cumulative=False):
        """Computes the number of documents per year.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.documents_by_year().head()
           Year  Num Documents                    ID
        0  2010              3       [141, 142, 143]
        1  2011              2            [139, 140]
        2  2012              2            [137, 138]
        3  2013              4  [133, 134, 135, 136]
        4  2014              2            [131, 132]


        >>> rdf.documents_by_year(cumulative=True).head()
           Year  Num Documents                    ID
        0  2010              3       [141, 142, 143]
        1  2011              5            [139, 140]
        2  2012              7            [137, 138]
        3  2013             11  [133, 134, 135, 136]
        4  2014             13            [131, 132]

        """
        result = self.documents_by_term(column="Year")
        years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
        result = result.set_index("Year")
        result = result.reindex(years, fill_value=0)
        result["ID"] = result.ID.map(
            lambda x: None if isinstance(x, int) and x == 0 else x
        )
        if cumulative is True:
            result["Num Documents"] = result["Num Documents"].cumsum()
        result = result.reset_index()

        return result

    def documents_by_term_by_year(
        self, column, sep=None, top_n=None, minmax_range=None
    ):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.documents_by_term_by_year(column='Author Keywords', top_n=5).head()
          Author Keywords  Year  Num Documents                                                 ID
        0   Deep learning  2018             20  [57, 61, 62, 66, 68, 70, 71, 73, 74, 76, 77, 8...
        1   Deep learning  2019             10             [3, 7, 13, 19, 25, 27, 34, 38, 40, 50]
        2            LSTM  2019              7                       [21, 30, 31, 34, 42, 45, 50]
        3   Deep Learning  2018              6                           [54, 78, 79, 86, 95, 97]
        4            LSTM  2018              5                               [71, 72, 74, 88, 89]

        >>> rdf.documents_by_term_by_year('Author Keywords',  minmax_range=(2,3)).head()
                     Author Keywords  Year  Num Documents               ID
        0                Forecasting  2018              3     [71, 95, 99]
        1   Recurrent neural network  2018              3     [74, 77, 83]
        2  Recurrent neural networks  2019              3     [16, 28, 32]
        3           Limit Order Book  2018              3     [78, 79, 82]
        4                       LSTM  2017              3  [112, 115, 121]

        >>> rdf.documents_by_term_by_year('Author Keywords',  top_n=3, minmax_range=(1,3), sep=';').head()
          Author Keywords  Year  Num Documents               ID
        0            LSTM  2017              3  [112, 115, 121]
        1   Deep learning  2017              2       [117, 120]
        2            LSTM  2013              2       [133, 135]
        3   Deep learning  2013              1            [134]
        4   Deep learning  2016              1            [125]

        """

        data = _expand(self[[column, "Year", "ID"]], column, sep)

        result = (
            data.groupby([column, "Year"], as_index=False)
            .agg({"ID": np.size})
            .rename(columns={"ID": "Num Documents"})
        )

        result = result.assign(
            ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
        )

        if top_n is not None:
            top_terms = self.documents_by_term(column, sep, top_n)[column].tolist()
            result = result[result[column].map(lambda x: x in top_terms)]

        if minmax_range is not None:
            min_val, max_val = minmax_range
            if min_val is not None:
                result = result[result["Num Documents"] >= min_val]
            if max_val is not None:
                result = result[result["Num Documents"] <= max_val]

        result.sort_values("Num Documents", ascending=False, inplace=True)

        result.index = list(range(len(result)))

        return result

    def co_ocurrence(
        self, column_r, column_c, sep_r=None, sep_c=None, minmax_range=None, top_n=None
    ):
        """Computes the number of rows containing two given items in different columns.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.co_ocurrence(column_r='Authors', sep_r=',', column_c='Document Type', top_n=5)
                Authors     Document Type  Num Documents                     ID
        0       Wang J.           Article              5  [3, 10, 80, 128, 128]
        1  Hernandez G.  Conference Paper              3          [52, 94, 100]
        2      Tefas A.  Conference Paper              3          [8, 110, 114]
        3       Wang J.  Conference Paper              2               [15, 87]
        4        Yan X.  Conference Paper              2               [13, 85]
        5      Zhang G.           Article              2              [27, 117]
        6      Zhang G.  Conference Paper              2              [78, 119]
        7        Yan X.           Article              1                   [44]

        """

        data = _expand(self[[column_r, column_c, "ID"]], column_r, sep_r)
        data = _expand(data[[column_r, column_c, "ID"]], column_c, sep_c)

        result = (
            data.groupby([column_r, column_c], as_index=False)
            .agg({"ID": np.size})
            .rename(columns={"ID": "Num Documents"})
        )

        result = result.assign(
            ID=data.groupby([column_r, column_c]).agg({"ID": list}).reset_index()["ID"]
        )

        if top_n is not None:

            top_terms_r = self.documents_by_term(column_r, sep_r, top_n)[
                column_r
            ].tolist()

            top_terms_c = self.documents_by_term(column_c, sep_c, top_n)[
                column_c
            ].tolist()

            result = result[
                result[column_r].map(lambda x: x in top_terms_r)
                & result[column_c].map(lambda x: x in top_terms_c)
            ]

        if minmax_range is not None:
            min_val, max_val = minmax_range
            if min_val is not None:
                result = result[result["Num Documents"] >= min_val]
            if max_val is not None:
                result = result[result["Num Documents"] <= max_val]

        result.sort_values("Num Documents", ascending=False, inplace=True)

        result.index = list(range(len(result)))

        return result

    def ocurrence(self, column, sep=None, top_n=None, minmax_range=None):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.ocurrence(column='Authors', sep=',', top_n=10)
           Authors (row) Authors (col)  Num Documents                                       ID
        0        Wang J.       Wang J.              9  [3, 10, 15, 80, 87, 128, 128, 128, 128]
        1       Zhang G.      Zhang G.              4                       [27, 78, 117, 119]
        2       Zhang Y.      Zhang Y.              3                              [4, 6, 109]
        3     Ar\xe9valo A.   Sandoval J.              3                            [52, 94, 100]
        8          Wu J.         Wu J.              3                            [34, 66, 115]
        9     Ar\xe9valo A.  Hernandez G.              3                            [52, 94, 100]
        10    Ar\xe9valo A.    Ar\xe9valo A.              3                            [52, 94, 100]
        12      Tefas A.  Iosifidis A.              3                            [8, 110, 114]
        15      Tefas A.      Tefas A.              3                            [8, 110, 114]
        18   Sandoval J.  Hernandez G.              3                            [52, 94, 100]
        19  Iosifidis A.  Iosifidis A.              3                            [8, 110, 114]
        22  Iosifidis A.      Tefas A.              3                            [8, 110, 114]
        32   Sandoval J.   Sandoval J.              3                            [52, 94, 100]
        34   Sandoval J.    Ar\xe9valo A.              3                            [52, 94, 100]
        35  Hernandez G.   Sandoval J.              3                            [52, 94, 100]
        36        Yan X.        Yan X.              3                             [13, 44, 85]
        37  Hernandez G.    Ar\xe9valo A.              3                            [52, 94, 100]
        38  Hernandez G.  Hernandez G.              3                            [52, 94, 100]

        """

        column_r = column + " (row)"
        column_c = column + " (col)"

        data = self[[column, "ID"]].copy()
        data.columns = [column_r, "ID"]
        data[column_c] = self[[column]]

        result = DataFrame(data).co_ocurrence(
            column_r=column_r,
            column_c=column_c,
            sep_r=sep,
            sep_c=sep,
            minmax_range=minmax_range,
            top_n=None,
        )

        if top_n is not None:

            top_terms = self.documents_by_term(column, sep, top_n)[column].tolist()

            result = result[
                result[column_r].map(lambda x: x in top_terms)
                & result[column_c].map(lambda x: x in top_terms)
            ]

        return result

    #
    #
    # Citation count
    #
    #

    def citations_by_term(self, column, sep=None, top_n=None, minmax_range=None):
        """Computes the number of citations by item in a column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.citations_by_term(column='Authors', sep=',', top_n=10)
                Authors  Cited by                             ID
        0   Hsieh T.-J.     188.0                          [140]
        1   Hsiao H.-F.     188.0                          [140]
        2     Yeh W.-C.     188.0                          [140]
        3  Hussain A.J.      52.0                     [125, 139]
        4     Krauss C.      49.0                           [62]
        5    Fischer T.      49.0                           [62]
        6       Wang J.      46.0  [3, 10, 15, 80, 87, 128, 128]
        7    Liatsis P.      42.0                          [139]
        8    Ghazali R.      42.0                          [139]
        9  Matsubara T.      37.0                          [124]

        >>> rdf.citations_by_term(column='Authors', sep=',', minmax_range=(30,50))
                   Authors  Cited by                             ID
        0        Krauss C.      49.0                           [62]
        1       Fischer T.      49.0                           [62]
        2          Wang J.      46.0  [3, 10, 15, 80, 87, 128, 128]
        3       Liatsis P.      42.0                          [139]
        4       Ghazali R.      42.0                          [139]
        5     Matsubara T.      37.0                          [124]
        6     Yoshihara A.      37.0                          [124]
        7        Uehara K.      37.0                          [124]
        8         Akita R.      37.0                          [124]
        9      Passalis N.      31.0                  [8, 110, 114]
        10      Gabbouj M.      31.0                  [8, 110, 114]
        11    Iosifidis A.      31.0                  [8, 110, 114]
        12        Tefas A.      31.0                  [8, 110, 114]
        13   Kanniainen J.      31.0                  [8, 110, 114]
        14  Tsantekidis A.      31.0                     [110, 114]

        """

        data = _expand(self[[column, "Cited by", "ID"]], column, sep)

        result = data.groupby(column, as_index=False).agg({"Cited by": np.sum})

        result = result.assign(
            ID=data.groupby(column).agg({"ID": list}).reset_index()["ID"]
        )
        result.sort_values("Cited by", ascending=False, inplace=True)

        if top_n is not None:
            result = result.head(top_n)

        if minmax_range is not None:
            min_val, max_val = minmax_range
            if min_val is not None:
                result = result[result["Cited by"] >= min_val]
            if max_val is not None:
                result = result[result["Cited by"] <= max_val]

        result.index = list(range(len(result)))

        return result

    def citations_by_year(self, cumulative=False):
        """Computes the number of citations by year.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.citations_by_year().head()
           Year  Cited by                    ID
        0  2010      21.0       [141, 142, 143]
        1  2011     230.0            [139, 140]
        2  2012      16.0            [137, 138]
        3  2013      36.0  [133, 134, 135, 136]
        4  2014      23.0            [131, 132]

        """

        result = self.citations_by_term(column="Year")
        years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
        result = result.set_index("Year")
        result = result.reindex(years, fill_value=0)
        result["ID"] = result.ID.map(
            lambda x: None if isinstance(x, int) and x == 0 else x
        )
        if cumulative is True:
            result["Cited by"] = result["Cited by"].cumsum()
        result = result.reset_index()

        return result

    def citations_by_term_by_year(
        self, column, sep=None, top_n=None, minmax_range=None
    ):
        """Computes the number of citations by term by year in a column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.citations_by_term_by_year('Authors', sep=',', top_n=5)
                Authors  Year  Cited by     ID
        0   Hsiao H.-F.  2011     188.0  [140]
        1   Hsieh T.-J.  2011     188.0  [140]
        2     Yeh W.-C.  2011     188.0  [140]
        3     Krauss C.  2018      49.0   [62]
        4  Hussain A.J.  2011      42.0  [139]
        5  Hussain A.J.  2016      10.0  [125]

        """

        data = _expand(self[[column, "Cited by", "Year", "ID"]], column, sep)

        result = data.groupby([column, "Year"], as_index=False).agg(
            {"Cited by": np.sum}
        )

        result = result.assign(
            ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
        )

        if top_n is not None:
            top_terms = self.citations_by_term(column, sep, top_n)[column].tolist()
            result = result[result[column].map(lambda x: x in top_terms)]

        if minmax_range is not None:
            min_val, max_val = minmax_range
            if min_val is not None:
                result = result[result["Cited by"] >= min_val]
            if max_val is not None:
                result = result[result["Cited by"] <= max_val]

        result.sort_values("Cited by", ascending=False, inplace=True)

        result.index = list(range(len(result)))

        return result
