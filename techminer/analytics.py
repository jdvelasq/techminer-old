"""
Analytics
==================================================================================================


"""

import pandas as pd
import numpy as np


def summary_by_year(df):
    """Computes the number of document and the number of total citations per year.
    This funciton adds the missing years in the sequence.

    
    Args:
        df (pandas.DataFrame): bibliographic dataframe.
        

    Returns:
        pandas.DataFrame.

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

    >>> summary_by_year(df)
        Year  Cited by  Num Documents     ID
    0  2010        21              2  [0, 1]
    1  2011        25              2  [2, 3]
    2  2012        14              1     [4]
    3  2013         0              0      []
    4  2014         0              0      []
    5  2015         0              0      []
    6  2016        15              1     [5]

    
    """
    data = df[["Year", "Cited by", "ID"]].explode("Year")
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
    result["Num Documents (Cum)"] = result["Num Documents"].cumsum()
    result["Cited by (Cum)"] = result["Cited by"].cumsum()
    result["Avg. Citations"] = result["Cited by"] / result["Num Documents"]
    result["Avg. Citations"] = result["Avg. Citations"].map(
        lambda x: 0 if pd.isna(x) else x
    )
    result = result.reset_index()
    return result


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
