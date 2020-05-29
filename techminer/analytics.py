"""
Analytics
==================================================================================================


"""

import pandas as pd
import numpy as np
from techminer.text import remove_acents, extract_country, extract_institution

scopus_to_wos_names = {
    "Abbreviated Source Title": "J9",
    "Abstract": "AB",
    "Access Type": "OA",
    "Affiliations": "C1",
    "Art. No.": "AR",
    "Author Keywords": "DE",
    "Author(s) ID": "RI",
    "Authors with affiliations": "AU_C1",
    "Authors": 'AU",
    "Cited by": "TC",
    "Cited references": "CR",
    "Author(s) Country": "AU_CO",
    "Author(s) Institution": "AU_IN",
    "DI": "DOI",
    "Document type": "DT",
    "Editors": "BE",
    "EID": "UT",
    "Index Keywords": "ID",
    "ISBN": "BN",
    "ISSN": "SN",
    "Issue": "IS",
    "Keywords": "KW",
    "Language of the Original Document": "LA",
    "Page count": "PG",
    "Page end": "EP",
    "Page start": "BP",
    "Publisher": "PU",
    "PubMed ID": "PM",
    "Source title": "SO",
    "Source": "FN",
    "Subject": "SC",
    "Title": "TI",
    "Volume": "VL",
    "Year": "PY",
}




def load_scopus(x):
    """Import filter for Scopus data.
    """
    #
    # 1. Rename and seleect columns
    #
    x = x.copy()
    x = x.rename(columns=scopus_to_wos_names)
    x = x[[w for w in x.columns if w in scopus_to_wos_names]]
    #
    # 2. Change ',' by ';' and remove '.' in author names
    #
    x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)
    if "AU" in x.columns:
        x["AU"] = x.Authors.map(
            lambda w: w.replace(",", ";").replace(".", "") if pd.isna(w) is False else w
        )
    #
    # Remove part of title in foreign language
    #
    if "TI" in x.columns:
        x["TI"] = x.TI.map(
            lambda w: w[0 : w.find("[")] if pd.isna(w) is False and w[-1] == "]" else w
        )
    #
    # Keywords fusion
    #
    author_keywords = x["DE"].map(
        lambda x: x.split(";") if x is not None else []
    )
    index_keywords = x["ID"].map(
        lambda x: x.split(";") if x is not None else []
    )
    keywords = author_keywords + index_keywords
    keywords = keywords.map(lambda w: [e for e in w if e != ""])
    keywords = keywords.map(lambda w: [e.strip() for e in w])
    keywords = keywords.map(lambda w: sorted(set(w)))
    keywords = keywords.map(lambda w: ";".join(w))
    keywords = keywords.map(lambda w: None if w == "" else w)
    x["KW"] = keywords
    #
    # Extract country and affiliation
    #
    if "C1" in x.columns:
        x["AU_CO"] = x.C1.map(lambda w: extract_country(w))
        x["AU_IN"] = x.C1.map(lambda w: extract_institution(w))
    #
    return x




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
