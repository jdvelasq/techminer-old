"""
TechMiner.DataFrame
==================================================================================================




"""


import pandas as pd

SCOPUS_SEPS = {"Authors": ",", "Author Keywords": ",", "Index Keywords": ","}


class DataFrame(pd.DataFrame):
    """Class to represent a dataframe of bibliographic records.
    """

    # ----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    # ----------------------------------------------------------------------------------------------
    def extract_terms(self, column, sep=None):
        """

        >>> import pandas as pd
        >>> pdf = pd.DataFrame({'A': ['1;2', '3', '3;4;5'], 'B':[0] * 3})
        >>> DataFrame(pdf).extract_terms(column='A', sep=';')
        array(['1', '2', '3', '4', '5'], dtype=object)

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'yyy', 'xxx, zzz', 'xxx, yyy, zzz']})
        >>> DataFrame(pdf).extract_terms(column='Authors')
        array(['xxx', 'yyy', ' zzz', ' yyy'], dtype=object)

        """
        result = self[column]
        if sep is None and column in SCOPUS_SEPS.keys():
            sep = SCOPUS_SEPS[column]
        if sep is not None:
            result = result.map(lambda x: x.split(sep))
            result = result.explode()
        return result.unique()

    # ----------------------------------------------------------------------------------------------
    def count_terms(self, column, sep=None):
        """

        >>> import pandas as pd
        >>> pdf = pd.DataFrame({'A': ['1;2', '3', '3;4;5'], 'B':[0] * 3})
        >>> DataFrame(pdf).count_terms(column='A', sep=';')
        5

        >>> pdf = pd.DataFrame({'Authors': ['xxx', 'yyy', 'xxx, zzz', 'xxx, yyy, zzz']})
        >>> DataFrame(pdf).count_terms(column='Authors')
        4


        """
        return len(self.extract_terms(column, sep))

