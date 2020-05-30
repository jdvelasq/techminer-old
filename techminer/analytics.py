"""
Analytics
==================================================================================================


"""

import pandas as pd
import numpy as np
from techminer.text import remove_accents, extract_country, extract_institution
from tqdm import tqdm


scopus_to_wos_names = {
    "Abbreviated Source Title": "J9",
    "Abstract": "AB",
    "Access Type": "OA",
    "Affiliations": "C1",
    "Art. No.": "AR",
    "Author Keywords": "DE",
    "Author(s) Country": "AU_CO",
    "Author(s) ID": "RI",
    "Author(s) Institution": "AU_IN",
    "Authors with affiliations": "AU_C1",
    "Authors": "AU",
    "Cited by": "TC",
    "Cited references": "CR",
    "DI": "DOI",
    "Document type": "DT",
    "Editors": "BE",
    "EID": "UT",
    "Index Keywords": "ID",
    "ISBN": "BN",
    "ISSN": "SN",
    "Issue": "IS",
    "Keywords": "KW",
    "Language of Original Document": "LA",
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

column_names_for_tables = {
    "AU": "Authors",
    "DE": "Author Keywords",
    "ID": "Index Keywords",
    "PY": "Year",
    "TC": "Times Cited",
}


def load_scopus(x):
    """Import filter for Scopus data.
    """
    with tqdm(total=5) as pbar:
        #
        # 1. Rename and seleect columns
        #
        x = x.copy()
        x = x.rename(columns=scopus_to_wos_names)
        x = x[[w for w in x.columns if w in scopus_to_wos_names.values()]]
        pbar.update(1)
        #
        # 2. Change ',' by ';' and remove '.' in author names
        #
        x = x.applymap(lambda w: remove_accents(w) if isinstance(w, str) else w)
        if "AU" in x.columns:
            x["AU"] = x.AU.map(
                lambda w: w.replace(",", ";").replace(".", "")
                if pd.isna(w) is False
                else w
            )
        pbar.update(1)
        #
        # 3. Remove part of title in foreign language
        #
        if "TI" in x.columns:
            x["TI"] = x.TI.map(
                lambda w: w[0 : w.find("[")]
                if pd.isna(w) is False and w[-1] == "]"
                else w
            )
        pbar.update(1)
        #
        # 4. Keywords fusion
        #
        if "DE" in x.columns.tolist() and "ID" in x.columns.tolist():
            author_keywords = x["DE"].map(
                lambda x: x.split(";") if pd.isna(x) is False else []
            )
            index_keywords = x["ID"].map(
                lambda x: x.split(";") if pd.isna(x) is False else []
            )
            keywords = author_keywords + index_keywords
            keywords = keywords.map(lambda w: [e for e in w if e != ""])
            keywords = keywords.map(lambda w: [e.strip() for e in w])
            keywords = keywords.map(lambda w: sorted(set(w)))
            keywords = keywords.map(lambda w: ";".join(w))
            keywords = keywords.map(lambda w: None if w == "" else w)
            x["DE_ID"] = keywords
        pbar.update(1)
        #
        # 5. Extract country and affiliation
        #
        if "C1" in x.columns:

            x["AU_CO"] = x.C1.map(
                lambda w: extract_country(w) if pd.isna(w) is False else w
            )
            x["AU_IN"] = x.C1.map(
                lambda w: extract_institution(w) if pd.isna(w) is False else w
            )
        #
        # 6. Adds RecID
        #
        x["RecID"] = range(len(x))
        pbar.update(1)
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
    ...          "PY": [2010, 2010, 2011, 2011, 2012, 2016],
    ...          "TC": list(range(10,16)),
    ...          "RecID": list(range(6)),
    ...     }
    ... )
    >>> df
         PY  TC  RecID
    0  2010  10      0
    1  2010  11      1
    2  2011  12      2
    3  2011  13      3
    4  2012  14      4
    5  2016  15      5

    >>> summary_by_year(df)
       Year  Times Cited  ...  Times Cited (Cum) Avg. Times Cited
    0  2010           21  ...                 21             10.5
    1  2011           25  ...                 46             12.5
    2  2012           14  ...                 60             14.0
    3  2013            0  ...                 60              0.0
    4  2014            0  ...                 60              0.0
    5  2015            0  ...                 60              0.0
    6  2016           15  ...                 75             15.0
    <BLANKLINE>
    [7 rows x 7 columns]

    
    """
    data = df[["PY", "TC", "RecID"]].explode("PY")
    data["Num Documents"] = 1
    result = data.groupby("PY", as_index=False).agg(
        {"TC": np.sum, "Num Documents": np.size}
    )
    result = result.assign(
        RecID=data.groupby("PY").agg({"RecID": list}).reset_index()["RecID"]
    )
    result["TC"] = result["TC"].map(lambda x: int(x))
    years = [year for year in range(result.PY.min(), result.PY.max() + 1)]
    result = result.set_index("PY")
    result = result.reindex(years, fill_value=0)
    result["RecID"] = result["RecID"].map(lambda x: [] if x == 0 else x)
    result.sort_values(
        "PY", ascending=True, inplace=True,
    )
    result["Num Documents (Cum)"] = result["Num Documents"].cumsum()
    result["Times Cited (Cum)"] = result["TC"].cumsum()
    result["Avg. Times Cited"] = result["TC"] / result["Num Documents"]
    result["Avg. Times Cited"] = result["Avg. Times Cited"].map(
        lambda x: 0 if pd.isna(x) else x
    )
    result = result.reset_index()
    result = result.rename(columns=column_names_for_tables)
    return result


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
