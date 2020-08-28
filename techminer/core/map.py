import pandas as pd


from techminer.core.params import MULTIVALUED_COLS


def map_(x, column, f):
    """Applies function f to column in dataframe x.

    >>> import pandas as pd
    >>> x = pd.DataFrame({'Affiliations': ['USA; Russian Federation']})
    >>> map_(x, 'Affiliations', lambda w: __extract_country(w))
    0    United States;Russia
    Name: Affiliations, dtype: object


    """
    x = x.copy()
    if column in MULTIVALUED_COLS:
        z = x[column].map(lambda w: w.split(";") if not pd.isna(w) else w)
        z = z.map(lambda w: [f(z.strip()) for z in w] if isinstance(w, list) else w)
        z = z.map(
            lambda w: [z for z in w if not pd.isna(z)] if isinstance(w, list) else w
        )
        z = z.map(lambda w: ";".join(w) if isinstance(w, list) else w)
        return z
    return x[column].map(lambda w: f(w))
