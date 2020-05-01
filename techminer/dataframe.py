"""
TechMiner.DataFrame
==================================================================================================




"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .strings import asciify

SCOPUS_SEPS = {"Authors": ",", "Author Keywords": ";", "Index Keywords": ";", "ID": ";"}


def _expand(pdf, column, sep):

    result = pdf.copy()
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
        result.index = list(range(len(result)))

    return result


class DataFrame(pd.DataFrame):
    """Class to represent a dataframe of bibliographic records.
    """

    #
    # Compatitbility with pandas.DataFrame
    #

    @property
    def _constructor_expanddim(self):
        return self

    #
    # Document ID
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
    # Accents
    #
    def remove_accents(self):
        """Remove accents for all strings on a DataFrame
        """
        return DataFrame(
            self.applymap(lambda x: asciify(x) if isinstance(x, str) else x)
        )

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
                        Column  Number of items Coverage (%)
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

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'xxx, zzz', 'yyy', 'xxx, yyy, zzz']})
        >>> DataFrame(pdf).extract_terms(column='Authors')
          Authors
        0     xxx
        1     yyy
        2     zzz

        """
        result = _expand(self, column, sep)
        result = pd.unique(result[column].dropna())
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

    def count_report(self):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.count_report()
                    Column  Number of items
        0          Authors              407
        1     Source title              103
        2  Author Keywords              404
        3   Index Keywords              881

        """
        columns = ["Authors", "Source title", "Author Keywords", "Index Keywords"]
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

    def summarize_by_term(self, column, sep):
        """Auxiliary function

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf = rdf.remove_accents()
        >>> rdf.summarize_by_term('Authors', sep=None).head(5)
                 Authors  Num Documents  Cited by                             ID
        0        Wang J.              7        46  [3, 10, 15, 80, 87, 128, 128]
        1       Zhang G.              4        12             [27, 78, 117, 119]
        2     Gabbouj M.              3        31                  [8, 110, 114]
        3   Iosifidis A.              3        31                  [8, 110, 114]
        4  Kanniainen J.              3        31                  [8, 110, 114]

        """
        data = _expand(self[[column, "Cited by", "ID"]], column, sep)
        result = (
            data.groupby(column, as_index=False)
            .agg({"ID": np.size, "Cited by": np.sum})
            .rename(columns={"ID": "Num Documents"})
        )
        result = result.assign(
            ID=data.groupby(column).agg({"ID": list}).reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        result.sort_values(
            ["Num Documents", "Cited by", "Authors"],
            ascending=[False, False, True],
            inplace=True,
        )
        result.index = list(range(len(result)))
        return result

    def documents_by_term(self, column, sep=None):
        """Computes the number of documents per term in a given column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents()
        >>> rdf.documents_by_term('Authors').head(5)
                Authors  Num Documents                             ID
        0       Wang J.              7  [3, 10, 15, 80, 87, 128, 128]
        1      Zhang G.              4             [27, 78, 117, 119]
        2    Arevalo A.              3                  [52, 94, 100]
        3    Gabbouj M.              3                  [8, 110, 114]
        4  Hernandez G.              3                  [52, 94, 100]

        """

        result = self.summarize_by_term(column, sep)
        result.pop("Cited by")
        result.sort_values(
            ["Num Documents", "Authors"], ascending=[False, True], inplace=True
        )
        result.index = list(range(len(result)))
        return result

    def citations_by_term(self, column, sep=None):
        """Computes the number of citations by item in a column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.citations_by_term(column='Authors', sep=',').head(10)
                Authors  Cited by                             ID
        0   Hsiao H.-F.       188                          [140]
        1   Hsieh T.-J.       188                          [140]
        2     Yeh W.-C.       188                          [140]
        3  Hussain A.J.        52                     [125, 139]
        4    Fischer T.        49                           [62]
        5     Krauss C.        49                           [62]
        6       Wang J.        46  [3, 10, 15, 80, 87, 128, 128]
        7    Ghazali R.        42                          [139]
        8    Liatsis P.        42                          [139]
        9      Akita R.        37                          [124]

        """
        result = self.summarize_by_term(column, sep)
        result.pop("Num Documents")
        result.sort_values(
            ["Cited by", "Authors"], ascending=[False, True], inplace=True
        )
        result.index = list(range(len(result)))
        return result

    #
    #
    # Documents and citations by year
    #
    #

    def summarize_by_year(self, cumulative=False):
        """Auxiliary function

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.summarize_by_year(cumulative=False).head(5)
           Year  Cited by  Num Documents                    ID
        0  2010        21              3       [141, 142, 143]
        1  2011       230              2            [139, 140]
        2  2012        16              2            [137, 138]
        3  2013        36              4  [133, 134, 135, 136]
        4  2014        23              2            [131, 132]

        >>> rdf.summarize_by_year(cumulative=True).head(5)
           Year  Cited by  Num Documents                    ID
        0  2010        21              3       [141, 142, 143]
        1  2011       251              5            [139, 140]
        2  2012       267              7            [137, 138]
        3  2013       303             11  [133, 134, 135, 136]
        4  2014       326             13            [131, 132]

        """
        data = _expand(self[["Year", "Cited by", "ID"]], "Year", None)

        result = (
            data.groupby("Year", as_index=False)
            .agg({"Cited by": np.sum, "ID": np.size})
            .rename(columns={"ID": "Num Documents"})
        )

        result = result.assign(
            ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
        )

        result["Cited by"] = result["Cited by"].map(lambda x: int(x))

        years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
        result = result.set_index("Year")
        result = result.reindex(years, fill_value=0)

        result.sort_values("Year", ascending=True, inplace=True)

        if cumulative is True:
            result["Num Documents"] = result["Num Documents"].cumsum()
            result["Cited by"] = result["Cited by"].cumsum()

        result = result.reset_index()

        return result

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
        result = self.summarize_by_year(cumulative)
        result.pop("Cited by")
        result.index = list(range(len(result)))
        return result

    def citations_by_year(self, cumulative=False):
        """Computes the number of citations by year.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf.citations_by_year().head()
           Year  Cited by                    ID
        0  2010        21       [141, 142, 143]
        1  2011       230            [139, 140]
        2  2012        16            [137, 138]
        3  2013        36  [133, 134, 135, 136]
        4  2014        23            [131, 132]

        """
        result = self.summarize_by_year(cumulative)
        result.pop("Num Documents")
        result.index = list(range(len(result)))
        return result

    # #
    # #
    # #  Documents and citations by term per year
    # #
    # #

    # def _docs_and_citedby_by_term_per_year(self, column, sep=None):

    #     data = _expand(self[[column, "Year", "Cited by", "ID"]], column, sep)

    #     result = (
    #         data.groupby([column, "Year"], as_index=False)
    #         .agg({"Cited by": np.sum, "ID": np.size})
    #         .rename(columns={"ID": "Num Documents"})
    #     )

    #     result = result.assign(
    #         ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
    #     )

    #     result.sort_values("Year", ascending=True, inplace=True)

    #     result.index = list(range(len(result)))

    #     return result

    # def documents_by_term_per_year(
    #     self, column, sep=None, top_n=None, minmax_range=None
    # ):
    #     """

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.documents_by_term_per_year(column='Authors').head()
    #                                  Authors          Year  Num Documents     ID
    #     0                    Song Y. [2, 10]  2010 [3, 21]              1  [143]
    #     1                    Laws J. [1, 12]  2010 [3, 21]              1  [142]
    #     2                     Lin X. [2, 14]  2010 [3, 21]              1  [143]
    #     3                     Zhai F. [1, 9]  2010 [3, 21]              1  [143]
    #     4  [No author name available] [1, 0]  2010 [3, 21]              1  [141]

    #     >>> rdf.documents_by_term_per_year('Authors',  minmax_range=(2,3)).head()
    #                       Authors            Year  Num Documents          ID
    #     0         Wang J. [7, 46]    2016 [5, 85]              2  [128, 128]
    #     1    Iosifidis A. [3, 31]  2017 [19, 137]              2  [110, 114]
    #     2   Kanniainen J. [3, 31]  2017 [19, 137]              2  [110, 114]
    #     3      Gabbouj M. [3, 31]  2017 [19, 137]              2  [110, 114]
    #     4  Tsantekidis A. [2, 31]  2017 [19, 137]              2  [110, 114]

    #     >>> rdf.documents_by_term_per_year('Authors', minmax_range=(1,3)).head()
    #                                  Authors          Year  Num Documents     ID
    #     0                    Song Y. [2, 10]  2010 [3, 21]              1  [143]
    #     1                    Laws J. [1, 12]  2010 [3, 21]              1  [142]
    #     2                     Lin X. [2, 14]  2010 [3, 21]              1  [143]
    #     3                     Zhai F. [1, 9]  2010 [3, 21]              1  [143]
    #     4  [No author name available] [1, 0]  2010 [3, 21]              1  [141]

    #     """

    #     result = self._docs_and_citedby_by_term_per_year(column, sep)

    #     if top_n is not None:

    #         top_terms = self._docs_and_citedby_by_term(column, sep)
    #         top_terms.sort_values("Cited by", ascending=False, inplace=True)
    #         top_terms = top_terms.head(top_n)[column]
    #         result = result[result[column].map(lambda x: x in top_terms)]

    #     if minmax_range is not None:
    #         min_val, max_val = minmax_range
    #         if min_val is not None:
    #             result = result[result["Num Documents"] >= min_val]
    #         if max_val is not None:
    #             result = result[result["Num Documents"] <= max_val]

    #     result.sort_values(
    #         ["Year", "Num Documents"], ascending=[True, False], inplace=True
    #     )

    #     rename_years = self._docs_and_citedby_by_year(cumulative=False)
    #     rename_years = {
    #         year: "{:d} [{:d}, {:d}]".format(year, docs_per_year, cites_per_year)
    #         for year, docs_per_year, cites_per_year in zip(
    #             rename_years["Year"],
    #             rename_years["Num Documents"],
    #             rename_years["Cited by"],
    #         )
    #     }
    #     result["Year"] = result["Year"].map(lambda y: rename_years[y])

    #     rename_terms = self._docs_and_citedby_by_term(column, sep)
    #     rename_terms = {
    #         term: "{:s} [{:d}, {:d}]".format(term, docs_per_term, cites_per_term)
    #         for term, docs_per_term, cites_per_term in zip(
    #             rename_terms[column],
    #             rename_terms["Num Documents"],
    #             rename_terms["Cited by"],
    #         )
    #     }
    #     result[column] = result[column].map(lambda t: rename_terms[t])

    #     result.pop("Cited by")

    #     result.index = list(range(len(result)))

    #     return result

    # def citations_by_term_per_year(
    #     self, column, sep=None, top_n=None, minmax_range=None
    # ):
    #     """Computes the number of citations by term by year in a column.

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.citations_by_term_per_year('Authors').head(5)
    #                     Authors          Year  Cited by     ID
    #     0       Laws J. [1, 12]  2010 [3, 21]      12.0  [142]
    #     1    Dunis C.L. [1, 12]  2010 [3, 21]      12.0  [142]
    #     2  Sermpinis G. [1, 12]  2010 [3, 21]      12.0  [142]
    #     3       Song Y. [2, 10]  2010 [3, 21]       9.0  [143]
    #     4        Lin X. [2, 14]  2010 [3, 21]       9.0  [143]

    #     """

    #     result = self._docs_and_citedby_by_term_per_year(column, sep)

    #     if top_n is not None:

    #         top_terms = self._docs_and_citedby_by_term(column, sep)
    #         top_terms.sort_values("Num Documents", ascending=False, inplace=True)
    #         top_terms = top_terms.head(top_n)[column]
    #         result = result[result[column].map(lambda x: x in top_terms)]

    #     if minmax_range is not None:
    #         min_val, max_val = minmax_range
    #         if min_val is not None:
    #             result = result[result["Cited by"] >= min_val]
    #         if max_val is not None:
    #             result = result[result["Cited by"] <= max_val]

    #     result.sort_values(["Year", "Cited by"], ascending=[True, False], inplace=True)

    #     rename_years = self._docs_and_citedby_by_year(cumulative=False)
    #     rename_years = {
    #         year: "{:d} [{:d}, {:d}]".format(year, docs_per_year, cites_per_year)
    #         for year, docs_per_year, cites_per_year in zip(
    #             rename_years["Year"],
    #             rename_years["Num Documents"],
    #             rename_years["Cited by"],
    #         )
    #     }
    #     result["Year"] = result["Year"].map(lambda y: rename_years[y])

    #     rename_terms = self._docs_and_citedby_by_term(column, sep)
    #     rename_terms = {
    #         term: "{:s} [{:d}, {:d}]".format(term, docs_per_term, cites_per_term)
    #         for term, docs_per_term, cites_per_term in zip(
    #             rename_terms[column],
    #             rename_terms["Num Documents"],
    #             rename_terms["Cited by"],
    #         )
    #     }
    #     result[column] = result[column].map(lambda t: rename_terms[t])

    #     result.pop("Num Documents")

    #     result.index = list(range(len(result)))

    #     return result

    # def documents_by_terms_per_terms_per_year(
    #     self, column_r, column_c, sep_r=None, sep_c=None, top_n=None, minmax_range=None
    # ):
    #     """

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.documents_by_terms_per_terms_per_year(
    #     ... column_r='Authors', sep_r=',', column_c='Author Keywords', sep_c=';', top_n=5)
    #             Authors        Author Keywords  Year  Num Documents         ID
    #     0  Hernandez G.          Deep learning  2018              2  [94, 100]
    #     1       Wang J.          Deep Learning  2019              1       [15]
    #     2       Wang J.          Deep learning  2018              1       [87]
    #     3       Wang J.          Deep learning  2019              1        [3]
    #     4        Yan X.          Deep learning  2019              1       [13]
    #     5        Yan X.  Financial time series  2019              1       [13]
    #     6      Zhang G.          Deep Learning  2018              1       [78]
    #     7      Zhang G.          Deep learning  2017              1      [117]
    #     8      Zhang G.          Deep learning  2019              1       [27]
    #     9      Zhang G.  Financial time series  2017              1      [119]

    #     """

    #     data = _expand(self[[column_r, column_c, "Year", "ID"]], column_r, sep_r)
    #     data = _expand(data[[column_r, column_c, "Year", "ID"]], column_c, sep_c)

    #     result = (
    #         data.groupby([column_r, column_c, "Year"], as_index=False)
    #         .agg({"ID": np.size})
    #         .rename(columns={"ID": "Num Documents"})
    #     )

    #     result = result.assign(
    #         ID=data.groupby([column_r, column_c, "Year"])
    #         .agg({"ID": list})
    #         .reset_index()["ID"]
    #     )

    #     if top_n is not None:

    #         top_terms_r = self.documents_by_term(column_r, sep_r, top_n)[
    #             column_r
    #         ].tolist()

    #         top_terms_c = self.documents_by_term(column_c, sep_c, top_n)[
    #             column_c
    #         ].tolist()

    #         result = result[
    #             result[column_r].map(lambda x: x in top_terms_r)
    #             & result[column_c].map(lambda x: x in top_terms_c)
    #         ]

    #     if minmax_range is not None:
    #         min_val, max_val = minmax_range
    #         if min_val is not None:
    #             result = result[result["Num Documents"] >= min_val]
    #         if max_val is not None:
    #             result = result[result["Num Documents"] <= max_val]

    #     result.sort_values("Num Documents", ascending=False, inplace=True)

    #     result.index = list(range(len(result)))

    #     return result

    # def co_ocurrence(
    #     self, column_r, column_c, sep_r=None, sep_c=None, minmax_range=None, top_n=None
    # ):
    #     """Computes the number of rows containing two given items in different columns.

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.co_ocurrence(column_r='Authors', column_c='Document Type').head(10)
    #             Authors     Document Type  Num Documents                     ID
    #     0       Wang J.           Article              5  [3, 10, 80, 128, 128]
    #     1  Hernandez G.  Conference Paper              3          [52, 94, 100]
    #     2      Tefas A.  Conference Paper              3          [8, 110, 114]
    #     3       Wang J.  Conference Paper              2               [15, 87]
    #     4        Yan X.  Conference Paper              2               [13, 85]
    #     5      Zhang G.           Article              2              [27, 117]
    #     6      Zhang G.  Conference Paper              2              [78, 119]
    #     7        Yan X.           Article              1                   [44]

    #     >>> rdf.co_ocurrence(column_r='Authors', column_c='Document Type', top_n=5)

    #     """

    #     data = _expand(self[[column_r, column_c, "ID"]], column_r, sep_r)
    #     data = _expand(data[[column_r, column_c, "ID"]], column_c, sep_c)

    #     result = (
    #         data.groupby([column_r, column_c], as_index=False)
    #         .agg({"ID": np.size})
    #         .rename(columns={"ID": "Num Documents"})
    #     )

    #     result = result.assign(
    #         ID=data.groupby([column_r, column_c]).agg({"ID": list}).reset_index()["ID"]
    #     )

    #     if top_n is not None:

    #         top_terms_r = self._docs_and_citedby_by_term(column_r, sep_r)
    #         top_terms_r.sort_values("Cited by", ascending=False, inplace=True)
    #         top_terms_r = top_terms_r.head(top_n)
    #         top_terms_r = top_terms_r[column_r].tolist()

    #         top_terms_c = self._docs_and_citedby_by_term(column_c, sep_c)
    #         top_terms_c.sort_values("Cited by", ascending=False, inplace=True)
    #         top_terms_c = top_terms_c.head(top_n)
    #         top_terms_c = top_terms_c[column_c].tolist()

    #         result = result[
    #             result[column_r].map(lambda x: x in top_terms_r)
    #             & result[column_c].map(lambda x: x in top_terms_c)
    #         ]

    #     if minmax_range is not None:
    #         min_val, max_val = minmax_range
    #         if min_val is not None:
    #             result = result[result["Num Documents"] >= min_val]
    #         if max_val is not None:
    #             result = result[result["Num Documents"] <= max_val]

    #     result.sort_values(
    #         ["Num Documents", column_r, column_c],
    #         ascending=[False, True, True],
    #         inplace=True,
    #     )

    #     rename_terms_r = self._docs_and_citedby_by_term(column_r, sep_r)
    #     rename_terms_r = {
    #         term: "{:s} [{:d}, {:d}]".format(term, docs_per_term, cites_per_term)
    #         for term, docs_per_term, cites_per_term in zip(
    #             rename_terms_r[column_r],
    #             rename_terms_r["Num Documents"],
    #             rename_terms_r["Cited by"],
    #         )
    #     }
    #     result[column_r] = result[column_r].map(lambda t: rename_terms_r[t])

    #     rename_terms_c = self._docs_and_citedby_by_term(column_c, sep_c)
    #     rename_terms_c = {
    #         term: "{:s} [{:d}, {:d}]".format(term, docs_per_term, cites_per_term)
    #         for term, docs_per_term, cites_per_term in zip(
    #             rename_terms_c[column_c],
    #             rename_terms_c["Num Documents"],
    #             rename_terms_c["Cited by"],
    #         )
    #     }
    #     result[column_c] = result[column_c].map(lambda t: rename_terms_c[t])

    #     result.index = list(range(len(result)))

    #     return result

    # def ocurrence(self, column, sep=None, top_n=None, minmax_range=None):
    #     """

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.ocurrence(column='Authors', sep=',', top_n=10)
    #        Authors (row) Authors (col)  Num Documents                                       ID
    #     0        Wang J.       Wang J.              9  [3, 10, 15, 80, 87, 128, 128, 128, 128]
    #     1       Zhang G.      Zhang G.              4                       [27, 78, 117, 119]
    #     2       Zhang Y.      Zhang Y.              3                              [4, 6, 109]
    #     3     Ar\xe9valo A.   Sandoval J.              3                            [52, 94, 100]
    #     8          Wu J.         Wu J.              3                            [34, 66, 115]
    #     9     Ar\xe9valo A.  Hernandez G.              3                            [52, 94, 100]
    #     10    Ar\xe9valo A.    Ar\xe9valo A.              3                            [52, 94, 100]
    #     12      Tefas A.  Iosifidis A.              3                            [8, 110, 114]
    #     15      Tefas A.      Tefas A.              3                            [8, 110, 114]
    #     18   Sandoval J.  Hernandez G.              3                            [52, 94, 100]
    #     19  Iosifidis A.  Iosifidis A.              3                            [8, 110, 114]
    #     22  Iosifidis A.      Tefas A.              3                            [8, 110, 114]
    #     32   Sandoval J.   Sandoval J.              3                            [52, 94, 100]
    #     34   Sandoval J.    Ar\xe9valo A.              3                            [52, 94, 100]
    #     35  Hernandez G.   Sandoval J.              3                            [52, 94, 100]
    #     36        Yan X.        Yan X.              3                             [13, 44, 85]
    #     37  Hernandez G.    Ar\xe9valo A.              3                            [52, 94, 100]
    #     38  Hernandez G.  Hernandez G.              3                            [52, 94, 100]

    #     """

    #     column_r = column + " (row)"
    #     column_c = column + " (col)"

    #     data = self[[column, "ID"]].copy()
    #     data.columns = [column_r, "ID"]
    #     data[column_c] = self[[column]]

    #     result = DataFrame(data).co_ocurrence(
    #         column_r=column_r,
    #         column_c=column_c,
    #         sep_r=sep,
    #         sep_c=sep,
    #         minmax_range=minmax_range,
    #         top_n=None,
    #     )

    #     if top_n is not None:

    #         top_terms = self.documents_by_term(column, sep, top_n)[column].tolist()

    #         result = result[
    #             result[column_r].map(lambda x: x in top_terms)
    #             & result[column_c].map(lambda x: x in top_terms)
    #         ]

    #     return result

    # #
    # #
    # # Citation count
    # #
    # #

    # #
    # #
    # #  Analytical functions
    # #
    # #

    # def compute_tfm(self, column, sep, top_n=20):
    #     """

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.compute_tfm('Authors', sep=',', top_n=5).head() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    #        Hernandez G.  Tefas A.  Wang J.  Yan X.  Zhang G.
    #     0             0         0        0       0         0
    #     1             0         0        0       0         0
    #     2             0         0        0       0         0
    #     3             0         0        1       0         0
    #     4             0         0        0       0         0

    #     """

    #     data = self[[column, "ID"]].copy()
    #     data["value"] = 1.0
    #     data = _expand(data, column, sep)

    #     result = pd.pivot_table(
    #         data=data, index="ID", columns=column, margins=False, fill_value=0.0,
    #     )
    #     result.columns = [b for _, b in result.columns]

    #     if top_n is not None:
    #         top_terms = self.documents_by_term(column, sep, top_n)[column].tolist()
    #         result = result[
    #             result.columns[[col in top_terms for col in result.columns]]
    #         ]

    #     result.index = list(range(len(result)))

    #     return result

    # def auto_corr(self, column, sep=None, top_n=20, method="pearson"):
    #     """Computes the autocorrelation among items in a column of the dataframe.

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.auto_corr(column='Authors', sep=',', top_n=5) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    #                   Hernandez G.  Tefas A.   Wang J.    Yan X.  Zhang G.
    #     Hernandez G.      1.000000 -0.021277 -0.030415 -0.021277 -0.024656
    #     Tefas A.         -0.021277  1.000000 -0.030415 -0.021277 -0.024656
    #     Wang J.          -0.030415 -0.030415  1.000000 -0.030415 -0.035245
    #     Yan X.           -0.021277 -0.021277 -0.030415  1.000000 -0.024656
    #     Zhang G.         -0.024656 -0.024656 -0.035245 -0.024656  1.000000

    #     """

    #     result = self.compute_tfm(column=column, sep=sep, top_n=top_n)

    #     if top_n is not None:
    #         top_terms = self.documents_by_term(column, sep, top_n)[column].tolist()
    #         result = result[
    #             result.columns[[col in top_terms for col in result.columns]]
    #         ]

    #     result = result.corr(method=method)

    #     return result

    # def cross_corr(
    #     self,
    #     column_r,
    #     column_c=None,
    #     sep_r=None,
    #     sep_c=None,
    #     top_n=8,
    #     method="pearson",
    # ):
    #     """Computes the crosscorrelation among items in two different columns of the dataframe.

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.cross_corr(column_r='Authors', sep_r=',', column_c='Author Keywords', sep_c=';', top_n=5)
    #                   Deep Learning  Deep learning  Financial time series      LSTM  Recurrent neural network
    #     Hernandez G.      -0.046635       0.020874              -0.038514 -0.064886                 -0.041351
    #     Tefas A.          -0.046635       0.020874              -0.038514  0.084112                 -0.041351
    #     Wang J.           -0.060710       0.149704               0.127494 -0.084469                 -0.053830
    #     Yan X.            -0.046635      -0.096780              -0.038514 -0.064886                 -0.041351
    #     Zhang G.          -0.054074      -0.112217              -0.044658  0.054337                 -0.047946

    #     """

    #     tfm_r = self.compute_tfm(column=column_r, sep=sep_r, top_n=top_n)
    #     tfm_c = self.compute_tfm(column=column_c, sep=sep_c, top_n=top_n)

    #     if top_n is not None:

    #         top_terms_r = self.documents_by_term(column_r, sep_r, top_n)[
    #             column_r
    #         ].tolist()
    #         tfm_r = tfm_r[tfm_r.columns[[col in top_terms_r for col in tfm_r.columns]]]

    #         top_terms_c = self.documents_by_term(column_c, sep_c, top_n)[
    #             column_c
    #         ].tolist()
    #         tfm_c = tfm_c[tfm_c.columns[[col in top_terms_c for col in tfm_c.columns]]]

    #     tfm = pd.concat([tfm_c, tfm_r], axis=1)
    #     result = tfm.corr(method=method)
    #     result = result[result.columns[[col in top_terms_c for col in result.columns]]]
    #     result = result[result.index.map(lambda x: x in top_terms_r)]

    #     return result

    # def factor_analysis(self, column, sep=None, n_components=None, top_n=10):
    #     """Computes the matrix of factors for terms in a given column.

    #     >>> from techminer.datasets import load_test_cleaned
    #     >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
    #     >>> rdf.factor_analysis(
    #     ...    column='Authors',
    #     ...    sep=',',
    #     ...    n_components=5,
    #     ...    top_n=40).head() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    #                        F0        F1        F2        F3        F4
    #     Ar\xe9valo A.  -0.029542  0.489821  0.002071  0.023022  0.012202
    #     Borovykh A. -0.006824 -0.009375 -0.003671 -0.060149 -0.040898
    #     Bu H.       -0.008162 -0.012594  0.099892 -0.066203  0.469863
    #     Chen J.     -0.008817 -0.014159  0.173627  0.092506  0.250997
    #     Chen Y.     -0.007328 -0.010763  0.064704 -0.057130  0.317516

    #     """

    #     tfm = self.compute_tfm(column, sep, top_n)
    #     terms = tfm.columns.tolist()

    #     if n_components is None:
    #         n_components = int(math.sqrt(len(set(terms))))

    #     pca = PCA(n_components=n_components)

    #     result = np.transpose(pca.fit(X=tfm.values).components_)
    #     result = pd.DataFrame(
    #         result, columns=["F" + str(i) for i in range(n_components)], index=terms
    #     )

    #     # cols = [["F" + str(i) for i in range(n_components)] for k in range(len(terms))]
    #     # rows = [[t for n in range(n_components)] for t in terms]
    #     # values = [values[i, j] for i in range(len(terms)) for j in range(n_components)]

    #     # cols = [e for row in cols for e in row]
    #     # rows = [e for row in rows for e in row]

    #     # result = pd.DataFrame({column: rows, "Factor": cols, "value": values})

    #     return result
