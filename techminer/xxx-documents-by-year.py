# def documents_by_year(self, cumulative=False):
#     """Computes the number of documents per year. This function adds the missing years in the sequence.

#     Args:
#         cumulative (bool): cumulate values per year.

#     Returns:
#         DataFrame.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#     ...          "Cited by": list(range(10,16)),
#     ...          "ID": list(range(6)),
#     ...     }
#     ... )
#     >>> df
#         Year  Cited by  ID
#     0  2010        10   0
#     1  2010        11   1
#     2  2011        12   2
#     3  2011        13   3
#     4  2012        14   4
#     5  2014        15   5

#     >>> DataFrame(df).documents_by_year()
#         Year  Num Documents      ID
#     0  2010              2  [0, 1]
#     1  2011              2  [2, 3]
#     2  2012              1     [4]
#     3  2013              0      []
#     4  2014              1     [5]

#     >>> DataFrame(df).documents_by_year(cumulative=True)
#         Year  Num Documents      ID
#     0  2010              2  [0, 1]
#     1  2011              4  [2, 3]
#     2  2012              5     [4]
#     3  2013              5      []
#     4  2014              6     [5]

#     """
#     result = self.summarize_by_year(cumulative)
#     result.pop("Cited by")
#     result = result.reset_index(drop=True)
#     return result
