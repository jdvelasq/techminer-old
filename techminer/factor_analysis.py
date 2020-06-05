"""
Factor analysis
==================================================================================================



"""


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from techminer.correlation import compute_tfm
from techminer.keywords import Keywords


def factor_analysis(x, column, n_components=None, as_matrix=True, keywords=None):
    """Computes the matrix of factors for terms in a given column.


    Args:
        column (str): the column to explode.
        sep (str): Character used as internal separator for the elements in the column.
        n_components: Number of components to compute.
        as_matrix (bool): the result is reshaped by melt or not.
        keywords (Keywords): filter the result using the specified Keywords object.

    Returns:
        DataFrame.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = [ 'A', 'A;B', 'B', 'A;B;C', 'B;D', 'A;B']
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
    1     A;B             a;b         1   1
    2       B               b         2   2
    3   A;B;C               c         3   3
    4     B;D             c;d         4   4
    5     A;B               d         5   5


    >>> compute_tfm(df, 'Authors')
       A  B  C  D
    0  1  0  0  0
    1  1  1  0  0
    2  0  1  0  0
    3  1  1  1  0
    4  0  1  0  1
    5  1  1  0  0


    >>> factor_analysis(df, 'Authors', n_components=3)
             F0            F1       F2
    A -0.774597 -0.000000e+00  0.00000
    B  0.258199  7.071068e-01 -0.57735
    C -0.258199  7.071068e-01  0.57735
    D  0.516398  1.110223e-16  0.57735

    >>> factor_analysis(df, 'Authors', n_components=3, as_matrix=False)
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

    >>> keywords = Keywords(['A', 'B', 'C'], ignore_case=False)
    >>> keywords = keywords.compile()
    >>> factor_analysis(df, 'Authors', n_components=3, keywords=keywords)
             F0        F1        F2
    A -0.888074  0.000000  0.459701
    B  0.325058  0.707107  0.627963
    C -0.325058  0.707107 -0.627963

    """

    tfm = compute_tfm(x, column, keywords)
    terms = tfm.columns.tolist()
    if n_components is None:
        n_components = int(np.sqrt(len(set(terms))))
    pca = PCA(n_components=n_components)
    result = np.transpose(pca.fit(X=tfm.values).components_)
    result = pd.DataFrame(
        result, columns=["F" + str(i) for i in range(n_components)], index=terms
    )

    if keywords is not None:
        if keywords._patterns is None:
            keywords = keywords.compile()
        new_index = [w for w in result.index if w in keywords]
        result = result.loc[new_index, :]

    if as_matrix is True:
        return result
    return (
        result.reset_index()
        .melt("index")
        .rename(columns={"index": column, "variable": "Factor"})
    )


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
