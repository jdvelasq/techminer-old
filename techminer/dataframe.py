"""
TechMiner.DataFrame
==================================================================================================




"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .strings import asciify

SCOPUS_SEPS = {
    "Authors": ",",
    "Author(s) ID": ";",
    "Author Keywords": ";",
    "Index Keywords": ";",
    "ID": ";",
}


def _expand(pdf, column, sep):
    """

    >>> from techminer.datasets import load_test_cleaned
    >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
    >>> result = _expand(rdf, 'Authors', sep=None)
    >>> result[result['Authors'].map(lambda x: 'Wang J.' in x) ][['Authors', 'Author(s) ID', 'ID']]
            Authors                                     Author(s) ID   ID
    11      Wang J.                          57207830408;57207828548    3
    36   Wang J.(1)                          57205691331;55946707000   10
    51   Wang J.(2)  15060001900;57209464952;57209470539;57209477365   15
    262  Wang J.(3)              57194399361;56380147600;37002500800   80
    282  Wang J.(4)  57204819270;56108513500;57206642524;57206677306   87
    312  Wang J.-J.              57203011511;57204046656;57204046789   92
    434  Wang J.(1)  56527464300;55946707000;42361194900;55286614500  128
    435  Wang J.(5)  56527464300;55946707000;42361194900;55286614500  128


    >>> result[result['Authors'] == 'Wang J.(1)'][['Authors', 'Author(s) ID', 'ID']]
            Authors                                     Author(s) ID   ID
    36   Wang J.(1)                          57205691331;55946707000   10
    434  Wang J.(1)  56527464300;55946707000;42361194900;55286614500  128

    >>> result[result['ID'] == 128][['Authors', 'Author(s) ID', 'ID']]
            Authors                                     Author(s) ID   ID
    432     Fang W.  56527464300;55946707000;42361194900;55286614500  128
    433      Niu H.  56527464300;55946707000;42361194900;55286614500  128
    434  Wang J.(1)  56527464300;55946707000;42361194900;55286614500  128
    435  Wang J.(5)  56527464300;55946707000;42361194900;55286614500  128

    >>> result[result['Authors'].map(lambda x: 'Zhang G.' in x) ][['Authors', 'Author(s) ID', 'ID']]
             Authors                                       Author(s) ID   ID
    92      Zhang G.                44062068800;57005856100;56949237700   27
    254  Zhang G.(1)  57202058986;56528648300;15077721900;5719383101...   78
    402  Zhang G.(2)                 56670483600;7404745474;56278752300  117
    407  Zhang G.(2)                 57197735758;7404745474;56670483600  119

    """

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
        return DataFrame(self)

    #
    # Distinc authors with same name
    #
    def disambiguate_authors(
        self,
        col_authors="Authors",
        sep_authors=None,
        col_ids="Author(s) ID",
        sep_ids=None,
    ):
        """

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...    {
        ...        'Authors': [
        ...               'xxx x, xxx x, yyy y',
        ...               'xxx x, yyy y, ddd d',
        ...               'ddd d',
        ...               'eee e',
        ...               None,
        ...               '[No author name available]'
        ...          ],
        ...        'Author(s) ID': [
        ...               '0;2; 1;',
        ...               '6;1; 3;',
        ...               '4',
        ...               '5',
        ...               None,
        ...               '[No author name available]'
        ...          ]
        ...    }
        ... )
        >>> df
                              Authors                Author(s) ID
        0         xxx x, xxx x, yyy y                     0;2; 1;
        1         xxx x, yyy y, ddd d                     6;1; 3;
        2                       ddd d                           4
        3                       eee e                           5
        4                        None                        None
        5  [No author name available]  [No author name available]

        >>> DataFrame(df).disambiguate_authors()
                              Authors                Author(s) ID
        0        xxx x,xxx x(1),yyy y                      0;2; 1
        1        xxx x(2),yyy y,ddd d                      6;1; 3
        2                    ddd d(1)                           4
        3                       eee e                           5
        4                        None                        None
        5  [No author name available]  [No author name available]

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf = rdf.remove_accents()
        >>> rdf[rdf['Authors'].map(lambda x: 'Wang J.' in x)][['Authors', 'Author(s) ID', 'ID']]
                                        Authors                                      Author(s) ID   ID
        3                       Cao J., Wang J.                          57207830408;57207828548;    3
        10                     Wang B., Wang J.                          57205691331;55946707000;   10
        15      Du J., Liu Q., Chen K., Wang J.  15060001900;57209464952;57209470539;57209477365;   15
        80             Dong Y., Wang J., Guo Z.              57194399361;56380147600;37002500800;   80
        87     Luo R., Zhang W., Xu X., Wang J.  57204819270;56108513500;57206642524;57206677306;   87
        92   Tsai Y.-C., Chen J.-H., Wang J.-J.              57203011511;57204046656;57204046789;   92
        128   Wang J., Wang J., Fang W., Niu H.  56527464300;55946707000;42361194900;55286614500;  128

        >>> rdf = rdf.disambiguate_authors()
        >>> rdf[rdf['Authors'].map(lambda x: 'Wang J.' in x)][['Authors', 'Author(s) ID', 'ID']]
                                          Authors                                     Author(s) ID   ID
        3                          Cao J.,Wang J.                          57207830408;57207828548    3
        10                     Wang B.,Wang J.(1)                          57205691331;55946707000   10
        15        Du J.,Liu Q.,Chen K.,Wang J.(2)  15060001900;57209464952;57209470539;57209477365   15
        80              Dong Y.,Wang J.(3),Guo Z.              57194399361;56380147600;37002500800   80
        87    Luo R.,Zhang W.(1),Xu X.,Wang J.(4)  57204819270;56108513500;57206642524;57206677306   87
        92       Tsai Y.-C.,Chen J.-H.,Wang J.-J.              57203011511;57204046656;57204046789   92
        128  Wang J.(5),Wang J.(1),Fang W.,Niu H.  56527464300;55946707000;42361194900;55286614500  128

        """

        if sep_authors is None:
            sep_authors = SCOPUS_SEPS[col_authors]

        if sep_ids is None:
            sep_ids = SCOPUS_SEPS[col_ids]

        self[col_ids] = self[col_ids].map(
            lambda x: x[:-1] if x is not None and x[-1] == ";" else x
        )

        data = self[[col_authors, col_ids]]
        data = data.dropna()

        data["*info*"] = [(a, b) for (a, b) in zip(data[col_authors], data[col_ids])]

        data["*info*"] = data["*info*"].map(
            lambda x: [
                (u.strip(), v.strip())
                for u, v in zip(x[0].split(sep_authors), x[1].split(sep_ids))
            ]
        )

        data = data[["*info*"]].explode("*info*")
        data.index = range(len(data))

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

        result[col_authors] = result[col_ids].map(
            lambda x: sep_authors.join([ids_names[w.strip()] for w in x.split(sep_ids)])
            if x is not None
            else x
        )

        return DataFrame(result)

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
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.count_report()
                    Column  Number of items
        0          Authors              434
        1     Author(s) ID              434
        2     Source title              103
        3  Author Keywords              404
        4   Index Keywords              881

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

    def summarize_by_term(self, column, sep):
        """Auxiliary function

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID()
        >>> rdf = rdf.remove_accents().disambiguate_authors()
        >>> rdf.summarize_by_term('Authors', sep=None).head(5)
               Authors  Num Documents  Cited by     ID
        0     Aadil F.              1         0   [28]
        1  Adam M.T.P.              1         6   [70]
        2   Afolabi D.              1         3  [108]
        3     Afzal S.              1         0   [28]
        4     Ahmed S.              1         0   [39]

        >>> result = rdf.summarize_by_term('Authors', sep=None)
        >>> result[result['Authors'].map(lambda x: 'Wang J.' in x)]
                Authors  Num Documents  Cited by         ID
        337     Wang J.              1         0        [3]
        338  Wang J.(1)              2        19  [10, 128]
        339  Wang J.(2)              1         0       [15]
        340  Wang J.(3)              1         4       [80]
        341  Wang J.(4)              1         4       [87]
        342  Wang J.(5)              1        19      [128]
        343  Wang J.-J.              1         1       [92]

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
            [column, "Num Documents", "Cited by"],
            ascending=[True, False, False],
            inplace=True,
        )

        result.index = list(range(len(result)))
        return result

    def documents_by_term(self, column, sep=None):
        """Computes the number of documents per term in a given column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> result = rdf.documents_by_term('Authors').head(5)
        >>> result
                 Authors  Num Documents             ID
        0     Arevalo A.              3  [52, 94, 100]
        1     Gabbouj M.              3  [8, 110, 114]
        2   Hernandez G.              3  [52, 94, 100]
        3   Iosifidis A.              3  [8, 110, 114]
        4  Kanniainen J.              3  [8, 110, 114]

        >>> result['lenIDs'] = result['ID'].map(lambda x: len(x))
        >>> sum(result['Num Documents']) == sum(result['lenIDs'])
        True

        """

        result = self.summarize_by_term(column, sep)
        result.pop("Cited by")
        result.sort_values(
            ["Num Documents", column], ascending=[False, True], inplace=True
        )
        result.sort_values(
            ["Num Documents", column], ascending=[False, True], inplace=True
        )
        result.index = list(range(len(result)))
        return result

    def citations_by_term(self, column, sep=None):
        """Computes the number of citations by item in a column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.citations_by_term(column='Authors', sep=',').head(10)
                Authors  Cited by          ID
        0   Hsiao H.-F.       188       [140]
        1   Hsieh T.-J.       188       [140]
        2     Yeh W.-C.       188       [140]
        3  Hussain A.J.        52  [125, 139]
        4    Fischer T.        49        [62]
        5     Krauss C.        49        [62]
        6    Ghazali R.        42       [139]
        7    Liatsis P.        42       [139]
        8      Akita R.        37       [124]
        9  Matsubara T.        37       [124]

        """
        result = self.summarize_by_term(column, sep)
        result.pop("Num Documents")
        result.sort_values(["Cited by", column], ascending=[False, True], inplace=True)
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

    #
    #
    #  Documents and citations by term per year
    #
    #

    def summarize_by_term_per_year(self, column, sep=None):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.summarize_by_term_per_year(column='Authors').head()
                Authors  Year  Cited by  Num Documents     ID
        0    Dunis C.L.  2010        12              1  [142]
        1       Laws J.  2010        12              1  [142]
        2     Lin X.(1)  2010         9              1  [143]
        3  Sermpinis G.  2010        12              1  [142]
        4    Song Y.(1)  2010         9              1  [143]

        """
        data = _expand(self[["Year", column, "Cited by", "ID"]], column, sep)
        result = (
            data.groupby([column, "Year"], as_index=False)
            .agg({"Cited by": np.sum, "ID": np.size})
            .rename(columns={"ID": "Num Documents"})
        )
        result = result.assign(
            ID=data.groupby([column, "Year"]).agg({"ID": list}).reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        result.sort_values(["Year", column], ascending=True, inplace=True)
        result.index = list(range(len(result)))
        return result

    def documents_by_term_per_year(
        self, column, sep=None, top_n=None, minmax_range=None
    ):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.documents_by_term_per_year(column='Authors').head()
                Authors  Year  Num Documents     ID
        0    Dunis C.L.  2010              1  [142]
        1       Laws J.  2010              1  [142]
        2     Lin X.(1)  2010              1  [143]
        3  Sermpinis G.  2010              1  [142]
        4    Song Y.(1)  2010              1  [143]

        """

        result = self.summarize_by_term_per_year(column, sep)
        result.pop("Cited by")
        result.sort_values(
            ["Year", "Num Documents", column],
            ascending=[True, False, True],
            inplace=True,
        )
        result.index = list(range(len(result)))
        return result

    def citations_by_term_per_year(
        self, column, sep=None, top_n=None, minmax_range=None
    ):
        """Computes the number of citations by term by year in a column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.citations_by_term_per_year('Authors').head(5)
                Authors  Year  Cited by     ID
        0    Dunis C.L.  2010        12  [142]
        1       Laws J.  2010        12  [142]
        2  Sermpinis G.  2010        12  [142]
        3     Lin X.(1)  2010         9  [143]
        4    Song Y.(1)  2010         9  [143]

        """

        result = self.summarize_by_term_per_year(column, sep)
        result.pop("Num Documents")
        result.sort_values(
            ["Year", "Cited by", column], ascending=[True, False, True], inplace=True,
        )
        result.index = list(range(len(result)))
        return result

    #
    #
    #  Documents and citations by term per term per year
    #
    #

    def summarize_by_term_per_term_per_year(
        self, column_r, column_c, sep_r=None, sep_c=None
    ):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.summarize_by_term_per_term_per_year('Authors', 'Author Keywords').head(5)
                Authors Author Keywords  Year  Cited by  Num Documents     ID
        0  Bekiros S.D.            LSTM  2013         2              1  [133]
        1  Bekiros S.D.     Time series  2013         2              1  [133]
        2  Bekiros S.D.   Time weighted  2013         2              1  [133]
        3  Bekiros S.D.           stock  2013         2              1  [133]
        4        Cai X.         Cluster  2013         5              1  [135]
        """

        data = _expand(
            self[[column_r, column_c, "Year", "Cited by", "ID"]], column_r, sep_r
        )
        data = _expand(
            data[[column_r, column_c, "Year", "Cited by", "ID"]], column_c, sep_c
        )
        result = (
            data.groupby([column_r, column_c, "Year"], as_index=False)
            .agg({"Cited by": np.sum, "ID": np.size})
            .rename(columns={"ID": "Num Documents"})
        )
        result = result.assign(
            ID=data.groupby([column_r, column_c, "Year"])
            .agg({"ID": list})
            .reset_index()["ID"]
        )
        result["Cited by"] = result["Cited by"].map(lambda x: int(x))
        result.sort_values(["Year", column_r, column_c,], ascending=True, inplace=True)
        result.index = list(range(len(result)))
        return result

    def documents_by_terms_per_terms_per_year(
        self, column_r, column_c, sep_r=None, sep_c=None
    ):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.documents_by_terms_per_terms_per_year(
        ... column_r='Authors', column_c='Author Keywords').head(10)
                Authors          Author Keywords  Year  Num Documents     ID
        0  Bekiros S.D.                     LSTM  2013              1  [133]
        1  Bekiros S.D.              Time series  2013              1  [133]
        2  Bekiros S.D.            Time weighted  2013              1  [133]
        3  Bekiros S.D.                    stock  2013              1  [133]
        4        Cai X.                  Cluster  2013              1  [135]
        5        Cai X.  Correlation coefficient  2013              1  [135]
        6        Cai X.                     LSTM  2013              1  [135]
        7        Cai X.              Time series  2013              1  [135]
        8        Lai G.                  Cluster  2013              1  [135]
        9        Lai G.  Correlation coefficient  2013              1  [135]

        """

        result = self.summarize_by_term_per_term_per_year(
            column_r, column_c, sep_r, sep_c
        )
        result.pop("Cited by")
        result.sort_values(
            ["Year", column_r, column_c], ascending=[True, True, True], inplace=True,
        )
        result.index = list(range(len(result)))
        return result

    def citations_by_terms_per_terms_per_year(
        self, column_r, column_c, sep_r=None, sep_c=None
    ):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.citations_by_terms_per_terms_per_year(
        ... column_r='Authors', column_c='Author Keywords').head(10)
                Authors          Author Keywords  Year  Num Documents     ID
        0  Bekiros S.D.                     LSTM  2013              1  [133]
        1  Bekiros S.D.              Time series  2013              1  [133]
        2  Bekiros S.D.            Time weighted  2013              1  [133]
        3  Bekiros S.D.                    stock  2013              1  [133]
        4        Cai X.                  Cluster  2013              1  [135]
        5        Cai X.  Correlation coefficient  2013              1  [135]
        6        Cai X.                     LSTM  2013              1  [135]
        7        Cai X.              Time series  2013              1  [135]
        8        Lai G.                  Cluster  2013              1  [135]
        9        Lai G.  Correlation coefficient  2013              1  [135]

        """

        result = self.summarize_by_term_per_term_per_year(
            column_r, column_c, sep_r, sep_c
        )
        result.pop("Cited by")
        result.sort_values(
            ["Year", column_r, column_c], ascending=[True, True, True], inplace=True,
        )
        result.index = list(range(len(result)))
        return result

    #
    #
    #  Co-ccurrence
    #
    #

    def summarize_co_ocurrence(self, column_r, column_c, sep_r=None, sep_c=None):
        """Summarize ocurrence and citations.

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

        >>> DataFrame(df).summarize_co_ocurrence(column_r='Authors', column_c='Author Keywords')
          Authors (row) Author Keywords (col)  Num Documents  Cited by      ID
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



        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().disambiguate_authors()
        >>> rdf.summarize_co_ocurrence(column_r='Authors', column_c='Document Type').head(10)
           Authors (row) Document Type (col)  Num Documents  Cited by     ID
        0       Aadil F.             Article              1         0   [28]
        1    Adam M.T.P.    Conference Paper              1         6   [70]
        2     Afolabi D.             Article              1         3  [108]
        3       Afzal S.             Article              1         0   [28]
        4       Ahmed S.             Article              1         0   [39]
        5       Akita R.    Conference Paper              1        37  [124]
        6     Aktas M.S.    Conference Paper              1         0   [21]
        7    Al-Askar H.             Article              1        10  [125]
        8  Al-Jumeily D.             Article              1        10  [125]
        9  Ali Mahmud S.             Article              1         3  [131]

        """

        def generate_pairs(w, v):
            if sep_r is not None:
                w = [x.strip() for x in w.split(sep_r)]
            else:
                w = [w]
            if sep_c is not None:
                v = [x.strip() for x in v.split(sep_c)]
            else:
                v = [v]
            result = []
            for idx0 in range(len(w)):
                for idx1 in range(len(v)):
                    result.append((w[idx0], v[idx1]))
            return result

        if sep_r is None and column_r in SCOPUS_SEPS:
            sep_r = SCOPUS_SEPS[column_r]

        if sep_c is None and column_c in SCOPUS_SEPS:
            sep_c = SCOPUS_SEPS[column_c]

        data = self.copy()
        data = data[[column_r, column_c, "Cited by", "ID"]]
        data["Num Documents"] = 1
        data["pairs"] = [
            generate_pairs(a, b) for a, b in zip(data[column_r], data[column_c])
        ]
        data = data[["pairs", "Num Documents", "Cited by", "ID"]]
        data = data.explode("pairs")

        result = data.groupby("pairs", as_index=False).agg(
            {"Cited by": np.sum, "Num Documents": np.sum, "ID": list}
        )

        result["Cited by"] = result["Cited by"].map(int)

        result[column_r + " (row)"] = result["pairs"].map(lambda x: x[0])
        result[column_c + " (col)"] = result["pairs"].map(lambda x: x[1])
        result.pop("pairs")

        result = result[
            [
                column_r + " (row)",
                column_c + " (col)",
                "Num Documents",
                "Cited by",
                "ID",
            ]
        ]

        result = result.sort_values([column_r + " (row)", column_c + " (col)"])

        return result

    def co_ocurrence(self, column_r, column_c, sep_r=None, sep_c=None):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> rdf.co_ocurrence(column_r='Authors', column_c='Document Type').head(10)
                 Authors (row)    Document Type (col)  Num Documents             ID
        14      Arevalo A. [3]  Conference Paper [88]              3  [52, 94, 100]
        99      Gabbouj M. [3]  Conference Paper [88]              3  [8, 110, 114]
        119   Hernandez G. [3]  Conference Paper [88]              3  [52, 94, 100]
        139   Iosifidis A. [3]  Conference Paper [88]              3  [8, 110, 114]
        150  Kanniainen J. [3]  Conference Paper [88]              3  [8, 110, 114]
        178        Leon D. [3]  Conference Paper [88]              3  [52, 94, 100]
        252        Nino J. [3]  Conference Paper [88]              3  [52, 94, 100]
        265    Passalis N. [3]  Conference Paper [88]              3  [8, 110, 114]
        291    Sandoval J. [3]  Conference Paper [88]              3  [52, 94, 100]
        323       Tefas A. [3]  Conference Paper [88]              3  [8, 110, 114]


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

        result = self.summarize_co_ocurrence(column_r, column_c, sep_r, sep_c)
        result.pop("Cited by")
        result.sort_values(
            [column_r + " (row)", column_c + " (col)", "Num Documents",],
            ascending=[True, True, False],
            inplace=True,
        )
        result.index = list(range(len(result)))

        new_names = generate_dic(column_c, sep_c)
        result[column_c + " (col)"] = result[column_c + " (col)"].map(
            lambda x: new_names[x]
        )

        new_names = generate_dic(column_r, sep_r)
        result[column_r + " (row)"] = result[column_r + " (row)"].map(
            lambda x: new_names[x]
        )

        result = result.sort_values(
            ["Num Documents", column_r + " (row)", column_c + " (col)"],
            ascending=[False, True, True],
        )

        return result

    def co_citation(self, column_r, column_c, sep_r=None, sep_c=None):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> rdf.co_citation(column_r='Authors', column_c='Document Type').head(10)
               Authors (row)     Document Type (col)  Cited by          ID
        0  Hsiao H.-F. [188]  Conference Paper [371]       188       [140]
        1  Hsieh T.-J. [188]  Conference Paper [371]       188       [140]
        2    Yeh W.-C. [188]  Conference Paper [371]       188       [140]
        3  Hussain A.J. [52]           Article [323]        52  [125, 139]
        4    Fischer T. [49]           Article [323]        49        [62]
        5     Krauss C. [49]           Article [323]        49        [62]
        6    Ghazali R. [42]           Article [323]        42       [139]
        7    Liatsis P. [42]           Article [323]        42       [139]
        8      Akita R. [37]  Conference Paper [371]        37       [124]
        9  Matsubara T. [37]  Conference Paper [371]        37       [124]

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

        result = self.summarize_co_ocurrence(column_r, column_c, sep_r, sep_c)
        result.pop("Num Documents")
        result.sort_values(
            ["Cited by", column_r + " (row)", column_c + " (col)",],
            ascending=[False, True, True,],
            inplace=True,
        )
        result.index = list(range(len(result)))

        new_names = generate_dic(column_c, sep_c)
        result[column_c + " (col)"] = result[column_c + " (col)"].map(
            lambda x: new_names[x]
        )

        new_names = generate_dic(column_r, sep_r)
        result[column_r + " (row)"] = result[column_r + " (row)"].map(
            lambda x: new_names[x]
        )

        return result

    #
    #
    #  Occurrence
    #
    #

    def summarize_ocurrence(self, column, sep=None):
        """Summarize ocurrence and citations.


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

        >>> DataFrame(df).summarize_ocurrence(column='Authors')
          Authors (row) Authors (col)  Num Documents  Cited by            ID
        0             A             A              4         7  [0, 1, 2, 4]
        1             A             B              2         6        [2, 4]
        2             A             C              1         4           [4]
        3             B             B              4        15  [2, 3, 4, 6]
        4             B             C              1         4           [4]
        5             B             D              1         6           [6]
        6             C             C              1         4           [4]
        7             D             D              2        11        [5, 6]


        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> rdf.summarize_ocurrence(column='Authors').head(10)
          Authors (row) Authors (col)  Num Documents  Cited by     ID
        0      Aadil F.      Aadil F.              1         0   [28]
        1      Aadil F.    Mehmood I.              1         0   [28]
        2      Aadil F.        Rho S.              1         0   [28]
        3   Adam M.T.P.   Adam M.T.P.              1         6   [70]
        4   Adam M.T.P.        Fan Z.              1         6   [70]
        5   Adam M.T.P.         Hu Z.              1         6   [70]
        6   Adam M.T.P.       Lutz B.              1         6   [70]
        7   Adam M.T.P.    Neumann D.              1         6   [70]
        8    Afolabi D.    Afolabi D.              1         3  [108]
        9    Afolabi D.    Guan S.-U.              1         3  [108]

        """

        def generate_pairs(w):
            w = [x.strip() for x in w.split(sep)]
            result = []
            for idx0 in range(len(w)):
                for idx1 in range(idx0, len(w)):
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

        result[column + " (row)"] = result["pairs"].map(lambda x: x[0])
        result[column + " (col)"] = result["pairs"].map(lambda x: x[1])
        result.pop("pairs")

        result = result[
            [column + " (row)", column + " (col)", "Num Documents", "Cited by", "ID"]
        ]

        result = result.sort_values([column + " (row)", column + " (col)"])

        return result

    def ocurrence(self, column, sep=None):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> rdf.ocurrence(column='Authors').head(10)
               Authors (row)     Authors (col)  Num Documents             ID
        0     Arevalo A. [3]    Arevalo A. [3]              3  [52, 94, 100]
        1     Arevalo A. [3]       Leon D. [3]              3  [52, 94, 100]
        2     Arevalo A. [3]   Sandoval J. [3]              3  [52, 94, 100]
        3     Gabbouj M. [3]    Gabbouj M. [3]              3  [8, 110, 114]
        4     Gabbouj M. [3]  Iosifidis A. [3]              3  [8, 110, 114]
        5   Hernandez G. [3]  Hernandez G. [3]              3  [52, 94, 100]
        6   Hernandez G. [3]   Sandoval J. [3]              3  [52, 94, 100]
        7   Iosifidis A. [3]  Iosifidis A. [3]              3  [8, 110, 114]
        8  Kanniainen J. [3]    Gabbouj M. [3]              3  [8, 110, 114]
        9  Kanniainen J. [3]  Iosifidis A. [3]              3  [8, 110, 114]

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

        column_r = column + " (row)"
        column_c = column + " (col)"

        result = self.summarize_ocurrence(column, sep)
        result.pop("Cited by")
        result.sort_values(
            ["Num Documents", column_r, column_c],
            ascending=[False, True, True],
            inplace=True,
        )
        result.index = list(range(len(result)))

        new_names = generate_dic(column, sep)
        result[column_c] = result[column_c].map(lambda x: new_names[x])
        result[column_r] = result[column_r].map(lambda x: new_names[x])

        return result

    #
    #
    #  Analytical functions
    #
    #

    def compute_tfm(self, column, sep):
        """

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> tfm = rdf.compute_tfm('Authors', sep=',')
        >>> authors = rdf.documents_by_term('Authors').head()['Authors']
        >>> authors = authors.tolist()
        >>> tfm[authors].head() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           Arevalo A.  Gabbouj M.  Hernandez G.  Iosifidis A.  Kanniainen J.
        0           0           0             0             0              0
        1           0           0             0             0              0
        2           0           0             0             0              0
        3           0           0             0             0              0
        4           0           0             0             0              0

        """

        data = self[[column, "ID"]].copy()
        data["value"] = 1.0
        data = _expand(data, column, sep)

        result = pd.pivot_table(
            data=data, index="ID", columns=column, margins=False, fill_value=0.0,
        )
        result.columns = [b for _, b in result.columns]

        result.index = list(range(len(result)))

        return result

    def auto_corr(self, column, sep=None, method="pearson"):
        """Computes the autocorrelation among items in a column of the dataframe.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> result = rdf.auto_corr(column='Authors', sep=',')
        >>> authors = rdf.documents_by_term('Authors').head()['Authors'].tolist()
        >>> authors
        ['Arevalo A.', 'Gabbouj M.', 'Hernandez G.', 'Iosifidis A.', 'Kanniainen J.']

        >>> result.loc[authors, authors]  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                       Arevalo A.  Gabbouj M.  Hernandez G.  Iosifidis A.  Kanniainen J.
        Arevalo A.       1.000000   -0.021277      1.000000     -0.021277      -0.021277
        Gabbouj M.      -0.021277    1.000000     -0.021277      1.000000       1.000000
        Hernandez G.     1.000000   -0.021277      1.000000     -0.021277      -0.021277
        Iosifidis A.    -0.021277    1.000000     -0.021277      1.000000       1.000000
        Kanniainen J.   -0.021277    1.000000     -0.021277      1.000000       1.000000

        """

        result = self.compute_tfm(column=column, sep=sep)
        result = result.corr(method=method)
        return result

    def cross_corr(
        self, column_r, column_c=None, sep_r=None, sep_c=None, method="pearson",
    ):
        """Computes the crosscorrelation among items in two different columns of the dataframe.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> keywords = rdf.documents_by_term('Author Keywords').head()['Author Keywords'].tolist()
        >>> keywords
        ['Deep learning', 'LSTM', 'Deep Learning', 'Recurrent neural network', 'Financial time series']

        >>> result = rdf.cross_corr(column_r='Authors', column_c='Author Keywords')
        >>> result[keywords].head()

                      Deep Learning  Deep learning  Financial time series      LSTM  Recurrent neural network
        Hernandez G.      -0.046635       0.020874              -0.038514 -0.064886                 -0.041351
        Tefas A.          -0.046635       0.020874              -0.038514  0.084112                 -0.041351
        Wang J.           -0.060710       0.149704               0.127494 -0.084469                 -0.053830
        Yan X.            -0.046635      -0.096780              -0.038514 -0.064886                 -0.041351
        Zhang G.          -0.054074      -0.112217              -0.044658  0.054337                 -0.047946

        """

        tfm_r = self.compute_tfm(column=column_r, sep=sep_r)
        tfm_c = self.compute_tfm(column=column_c, sep=sep_c)
        tfm = pd.concat([tfm_c, tfm_r], axis=1)
        return tfm.corr(method=method)

    def factor_analysis(self, column, sep=None, n_components=None):
        """Computes the matrix of factors for terms in a given column.

        >>> from techminer.datasets import load_test_cleaned
        >>> rdf = DataFrame(load_test_cleaned().data).generate_ID().remove_accents().disambiguate_authors()
        >>> rdf.factor_analysis(
        ...    column='Authors',
        ...    sep=',',
        ...    n_components=5).head() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                           F0        F1        F2        F3            F4
        Aadil F.    -0.003299 -0.004744 -0.004056  0.376626 -5.496588e-15
        Adam M.T.P. -0.002974 -0.004202 -0.003542 -0.008438 -2.642608e-02
        Afolabi D.  -0.002708 -0.003770 -0.003144 -0.004172  2.765398e-16
        Afzal S.    -0.003299 -0.004744 -0.004056  0.376626 -5.578519e-15
        Ahmed S.    -0.002708 -0.003770 -0.003144 -0.004172  2.247380e-16

        """

        tfm = self.compute_tfm(column, sep)
        terms = tfm.columns.tolist()

        if n_components is None:
            n_components = int(math.sqrt(len(set(terms))))

        pca = PCA(n_components=n_components)

        result = np.transpose(pca.fit(X=tfm.values).components_)
        result = pd.DataFrame(
            result, columns=["F" + str(i) for i in range(n_components)], index=terms
        )

        return result
