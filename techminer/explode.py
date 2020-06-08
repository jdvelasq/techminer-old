


# MULTIVALUED_COLS = [
#     "Affiliations",
#     "Author Keywords",
#     "Author(s) Country",
#     "Author(s) ID",
#     "Author(s) Institution",
#     "Authors with affiliations",
#     "Authors",
#     "Countries",
#     "Index Keywords",
#     "Institutions",
#     "Keywords",
#     "Author Keywords (Cleaned)",
#     "Index Keywords (Cleaned)",
#     #
#     "Affiliations_",
#     "Author Keywords_",
#     "Author(s) Country_",
#     "Author(s) ID_",
#     "Author(s) Institution_",
#     "Authors with affiliations_",
#     "Authors_",
#     "Countries_",
#     "Index Keywords_",
#     "Institutions_",
#     "Keywords_",
#     "Author Keywords (Cleaned)_",
#     "Index Keywords (Cleaned)_",
#     #
#     "_key1_",
#     "_key2_",
# ]


##
##
## Auxiliary Functions
##
##
# def __explode(x, column):
#     """Transform each element of a field to a row, reseting index values.

#     Args:
#         column (str): the column to explode.
#         sep (str): Character used as internal separator for the elements in the column.

#     Returns:
#         DataFrame. Exploded dataframe.

#     Examples
#     ----------------------------------------------------------------------------------------------

#     >>> import pandas as pd
#     >>> x = pd.DataFrame(
#     ...     {
#     ...         "Authors": "author 0;author 1;author 2,author 3,author 4".split(","),
#     ...         "ID": list(range(3)),
#     ...      }
#     ... )
#     >>> x
#                           Authors  ID
#     0  author 0;author 1;author 2   0
#     1                    author 3   1
#     2                    author 4   2

#     >>> __explode(x, 'Authors')
#         Authors  ID
#     0  author 0   0
#     1  author 1   0
#     2  author 2   0
#     3  author 3   1
#     4  author 4   2

#     """
#     if column in MULTIVALUED_COLS:
#         x = x.copy()
#         x[column] = x[column].map(
#             lambda w: sorted(list(set(w.split(";")))) if isinstance(w, str) else w
#         )
#         x = x.explode(column)
#         x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
#         x = x.reset_index(drop=True)
#     return x

