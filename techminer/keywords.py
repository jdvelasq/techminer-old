r"""
TechMiner.Keywords
==================================================================================================

This object contains a list of unique keywords (terms of interest).   


Regular expressions recipes
---------------------------------------------------------------------------------------------------

The following code exemplify some common cases using regular expressions.

>>> Keywords('111').extract_from_text('one two three four five') is None
True

* Partial match.

>>> Keywords('hre').extract_from_text('one two three four five')
'hre'


* **Word whole only**. `r'\b'` represents word boundaries.

>>> kyw = Keywords(r'\btwo\b', use_re=True)
>>> kyw.extract_from_text('one two three four five')
'two'

>>> kyw = Keywords(r"\b(TWO)\b", use_re=True)
>>> kyw.extract_from_text('one two three four five')
'two'


* **Case sensitive**.

>>> Keywords(r'\btwo\b', ignore_case=False, use_re=True).extract_from_text('one two three four five')
'two'

>>> Keywords(r"\bTWO\b", ignore_case=False, use_re=True).extract_from_text('one TWO three four five')
'TWO'

>>> Keywords(r"\bTWO\b", ignore_case=False, use_re=True).extract_from_text('one two three four five') is None
True

* **A word followed by other word**.

>>> Keywords(r'\btwo\Wthree\b', ignore_case=False, use_re=True).extract_from_text('one two three four five')
'two three'


* **Multiple white spaces**.

>>> Keywords(r"two\W+three", ignore_case=False, use_re=True).extract_from_text('one two   three four five')
'two   three'

* **A list of keywords**.

>>> Keywords([r"xxx", r"two", r"yyy"]).extract_from_text('one two three four five')
'two'


* **Adjacent terms but the order is unimportant**.

>>> Keywords(r"\bthree\W+two\b|\btwo\W+three\b", use_re=True).extract_from_text('one two three four five')
'two three'

* **Near words**.

Two words (`'two'`, `'four'`) separated by any other.

>>> Keywords(r"\btwo\W+\w+\W+four\b", use_re=True).extract_from_text('one two three four five')
'two three four'


Two words (`'two'`, `'five'`) separated by one, two or three unspecified words.

>>> Keywords(r"\btwo\W+(?:\w+\W+){1,3}?five", use_re=True).extract_from_text('one two three four five')
'two three four five'

* **Or operator**.

>>> Keywords(r"123|two", use_re=True).extract_from_text('one two three four five')
'two'

* **And operator**. One word followed by other at any word distance.

>>> Keywords(r"\btwo\W+(?:\w+\W+)+?five", use_re=True).extract_from_text('one two three four five')
'two three four five'



Functions in this module
---------------------------------------------------------------------------------------------------

"""
import json
import re
import string

import geopandas
import pandas as pd

from .strings import find_string, fingerprint, replace_string


class Keywords:
    """Creates a Keywords object used to find, extract or remove terms of interest from a string.

    Args:
        x (string, list of strings) : keyword of list of keywords.
        ignore_case (bool) :  Ignore string case.
        full_match (bool): match whole word?.
        use_re (bool): keywords as interpreted as regular expressions.

    Returns:
        Keywords object


    """

    def __init__(self, x=None, ignore_case=True, full_match=False, use_re=False):
        if x is None:
            self._keywords = None
        else:
            if isinstance(x, str):
                x = [x]
            self._keywords = sorted(list(set(x)))
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._use_re = use_re

    # --------------------------------------------------------------------------------------------------------
    @property
    def keywords(self):
        return self._keywords

    # --------------------------------------------------------------------------------------------------------
    def __contains__(self, x):
        """Implements in operator.

        >>> x = ['Big data', 'neural networks']
        >>> 'Big data' in Keywords(x)  # doctest: +NORMALIZE_WHITESPACE
        True
        >>> 'big data' in Keywords(x)  # doctest: +NORMALIZE_WHITESPACE
        True
        >>> 'deep learning' in Keywords(x)  # doctest: +NORMALIZE_WHITESPACE
        False
        >>> 'big data' in Keywords(x, ignore_case=False)  # doctest: +NORMALIZE_WHITESPACE
        False

        """
        if self.extract_from_text(x) is None:
            return False
        return True

    # --------------------------------------------------------------------------------------------------------
    def __len__(self):
        """Returns the number of keywords.

        >>> len(Keywords(['Big data', 'neural netoworks']))  # doctest: +NORMALIZE_WHITESPACE
        2
        """
        return len(self._keywords)

    # --------------------------------------------------------------------------------------------------------
    def __repr__(self):
        """String representation of the object.

        >>> Keywords(['Big data', 'neural networks'])  # doctest: +NORMALIZE_WHITESPACE
        [
          "Big data",
          "neural networks"
        ]
        ignore_case=True, full_match=False, use_re=False  

        """
        text = json.dumps(self._keywords, indent=2, sort_keys=True)
        text += "\nignore_case={}, full_match={}, use_re={}".format(
            self._ignore_case.__repr__(),
            self._full_match.__repr__(),
            self._use_re.__repr__(),
        )
        return text

    # --------------------------------------------------------------------------------------------------------
    def __str__(self):
        return self.__repr__()

    # --------------------------------------------------------------------------------------------------------
    def add_keywords(self, x, sep=None):
        """Adds new keywords x to list of current keywords.

        Args:
            x (string, list of strings): new keywords to be added.
        
        Returns:
            Nothing

        >>> kyw = Keywords()
        >>> kyw.add_keywords('ann')
        >>> kyw
        [
          "ann"
        ]
        ignore_case=True, full_match=False, use_re=False
        >>> kyw.add_keywords('RNN')
        >>> kyw
        [
          "RNN",
          "ann"
        ]
        ignore_case=True, full_match=False, use_re=False
        >>> kyw.add_keywords(['deep learning', 'fuzzy'])
        >>> kyw
        [
          "RNN",
          "ann",
          "deep learning",
          "fuzzy"
        ]
        ignore_case=True, full_match=False, use_re=False

        """
        if isinstance(x, str):
            x = [x]

        if isinstance(x, Keywords):
            x = x._keywords

        if isinstance(x, pd.Series):
            x = x.tolist()

        if sep is not None:
            x = [
                z.strip()
                for y in x
                if y is not None
                for z in y.split(sep)
                if z.strip() != ""
            ]
        else:
            x = [y.strip() for y in x if y is not None and y.strip() != ""]

        if self._keywords is None:
            self._keywords = sorted(list(set(x)))
        else:
            x.extend(self._keywords)
            self._keywords = sorted(list(set(x)))

    # --------------------------------------------------------------------------------------------------------
    def extract_from_text(self, x, sep=";"):
        r"""Returns a new string with the keywords in string x matching the list of keywords used to fit the model.

        >>> Keywords([r"xxx", r"two", r"yyy"]).extract_from_text('one two three four five')
        'two'

        The funcion allows the extraction of complex patterns using regular expresions (regex). 
        Detail information about regex sintax in Python can be obtained at https://docs.python.org/3/library/re.html#re-syntax.

        Args:
            x (string): A string object.
            
        Returns:
            String.

        """

        if x is None:
            return None

        result = []
        for keyword in self._keywords:

            y = find_string(
                pattern=keyword,
                x=x,
                ignore_case=self._ignore_case,
                full_match=self._full_match,
                use_re=self._use_re,
            )

            if y is not None:
                result.extend([y])

        if len(result):
            return sep.join(sorted(list(set(result))))

        return None

    # --------------------------------------------------------------------------------------------------------
    def remove_from_text(self, x):
        """Returns a string removing the strings that match a 
        list of keywords from x.

        Args:
            x (string): A string object.
            
        Returns:
            String.


        >>> Keywords('aaa').remove_from_text('1 aaa 2')
        '1  2'

        >>> Keywords('aaa').remove_from_text('1 2')
        '1 2'

        >>> Keywords('aaa').remove_from_text('1 aaa 2 1 2')
        '1  2 1 2'

        >>> Keywords(['aaa', 'bbb']).remove_from_text('1 aaa bbb 2 1 aaa 2')
        '1   2 1  2'

        """

        if x is None:
            return None

        for keyword in self._keywords:

            found_string = find_string(
                pattern=keyword,
                x=x,
                ignore_case=self._ignore_case,
                full_match=self._full_match,
                use_re=self._use_re,
            )

            if found_string is not None:

                x = replace_string(
                    pattern=found_string,
                    x=x,
                    repl="",
                    ignore_case=False,
                    full_match=False,
                    use_re=False,
                )

        return x

    # --------------------------------------------------------------------------------------------------------
    def delete_keyword(self, x):
        """Remove string x from the keywords list.
        """
        self._keywords.remove(x)

    # --------------------------------------------------------------------------------------------------------
    def common(self, x, sep=None):
        """Returns True if x is in keywords list.

        Args:
            x (string): A string object.
            
        Returns:
            Boolean.

        >>> kyw = Keywords(['ann', 'big data', 'deep learning'])
        >>> kyw.common('Big Data')
        True
        >>> kyw.common('Python')
        False
        >>> kyw.common('Python|R', sep='|')
        False
        >>> kyw.common('Python|big data', sep='|')
        True

        """

        def _common(x):
            if self.extract_from_text(x) is None:
                return False
            else:
                return True

        if sep is None:
            return _common(x)

        return any([_common(y.strip()) for y in x.split(sep)])

    # #--------------------------------------------------------------------------------------------------------
    # def complement(self, x, sep=None):
    #     """Returns False if x is not in keywords list.

    #     Args:
    #         x (string): A string object.

    #     Returns:
    #         Boolean.

    #     >>> kyw = Keywords(['ann', 'big data', 'deep learning'])
    #     >>> kyw.complement('Big Data')
    #     False
    #     >>> kyw.complement('Python')
    #     True
    #     >>> kyw.complement('Python|R')
    #     True
    #     >>> kyw.complement('Python|big data')
    #     False

    #     """
    #     def _complement(x):
    #         if self.extract_from(x) is None:
    #             return True
    #         else:
    #             return False

    #     if sep is None:
    #         return _complement(x)

    #     return any([_complement(y) for y in x.split(sep)])

    # --------------------------------------------------------------------------------------------------------
    # def _stemming(self, x):
    #     x = fingerprint(x)
    #     return [self.extract(z) for z in x.split()]
    # --------------------------------------------------------------------------------------------------------

    # def stemming_and(self, x):
    #     """
    #     >>> x = ['computer software', 'contol software', 'computing software']
    #     >>> Keywords().add_keywords(x).stemming_and('computer softare')
    #     True
    #     >>> Keywords().add_keywords(x).stemming_and('machine')
    #     False

    #     """
    #     z = self._stemming(x)
    #     z = [self.w for w in z if z is not None]
    #     return all(z)
    # --------------------------------------------------------------------------------------------------------

    # def stemming_any(self, x):
    #     z = self._stemming(x)
    #     z = [w for w in z if z is not None]
    #     return any(z)
