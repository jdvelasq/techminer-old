r"""
Keywords manipulation
==================================================================================================

This object contains a list of unique keywords (terms of interest).   


# Regular expressions recipes
# ---------------------------------------------------------------------------------------------------

# The following code exemplify some common cases using regular expressions.

# >>> Keywords('111').extract_from_text('one two three four five') is None
# True

# * Partial match.

# >>> Keywords('hre').extract_from_text('one two three four five')
# 'hre'


# * **Word whole only**. `r'\b'` represents word boundaries.

# >>> kyw = Keywords(r'\btwo\b', use_re=True)
# >>> kyw.extract_from_text('one two three four five')
# 'two'

# >>> kyw = Keywords(r"\b(TWO)\b", use_re=True)
# >>> kyw.extract_from_text('one two three four five')
# 'two'


# * **Case sensitive**.

# >>> Keywords(r'\btwo\b', ignore_case=False, use_re=True).extract_from_text('one two three four five')
# 'two'

# >>> Keywords(r"\bTWO\b", ignore_case=False, use_re=True).extract_from_text('one TWO three four five')
# 'TWO'

# >>> Keywords(r"\bTWO\b", ignore_case=False, use_re=True).extract_from_text('one two three four five') is None
# True

# * **A word followed by other word**.

# >>> Keywords(r'\btwo\Wthree\b', ignore_case=False, use_re=True).extract_from_text('one two three four five')
# 'two three'


# * **Multiple white spaces**.

# >>> Keywords(r"two\W+three", ignore_case=False, use_re=True).extract_from_text('one two   three four five')
# 'two   three'

# * **A list of keywords**.

# >>> Keywords([r"xxx", r"two", r"yyy"]).extract_from_text('one two three four five')
# 'two'


# * **Adjacent terms but the order is unimportant**.

# >>> Keywords(r"\bthree\W+two\b|\btwo\W+three\b", use_re=True).extract_from_text('one two three four five')
# 'two three'

# * **Near words**.

# Two words (`'two'`, `'four'`) separated by any other.

# >>> Keywords(r"\btwo\W+\w+\W+four\b", use_re=True).extract_from_text('one two three four five')
# 'two three four'


# Two words (`'two'`, `'five'`) separated by one, two or three unspecified words.

# >>> Keywords(r"\btwo\W+(?:\w+\W+){1,3}?five", use_re=True).extract_from_text('one two three four five')
# 'two three four five'

# * **Or operator**.

# >>> Keywords(r"123|two", use_re=True).extract_from_text('one two three four five')
# 'two'

# * **And operator**. One word followed by other at any word distance.

# >>> Keywords(r"\btwo\W+(?:\w+\W+)+?five", use_re=True).extract_from_text('one two three four five')
# 'two three four five'



Functions in this module
---------------------------------------------------------------------------------------------------

"""
import json
import re
import string

import pandas as pd

# from .strings import find_string, fingerprint, replace_string


class Keywords:
    """Creates a Keywords object used to find, extract or remove terms of interest from a string.
    """

    def __init__(
        self, keywords=None, sep=None, ignore_case=True, full_match=False, use_re=False
    ):
        """Creates a keywords object.

        Args:
            keywords (string, list of strings, techminer.Keywords): set of keywords to add.
            sep (character): separator character in string lists.
            ignore_case (bool) :  Ignore string case.
            full_match (bool): match whole word?.
            use_re (bool): keywords as interpreted as regular expressions.
            

        Returns:
            Keywords object

        """
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._keywords = None
        self._patterns = None
        self._use_re = use_re
        self.add_keywords(keywords=keywords, sep=sep)

    @property
    def keywords(self):
        return self._keywords

    def __repr__(self):
        """String representation of the object.

        Examples
        ----------------------------------------------------------------------------------------------
        
        >>> Keywords(['Big data', 'neural networks'])  # doctest: +NORMALIZE_WHITESPACE
        [
          "Big data",
          "neural networks"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=False

        """
        text = json.dumps(self._keywords, indent=2, sort_keys=True)
        text += "\nignore_case={}, full_match={}, use_re={}, compiled={}".format(
            self._ignore_case.__repr__(),
            self._full_match.__repr__(),
            self._use_re.__repr__(),
            self._patterns is not None,
        )
        return text

    def __str__(self):
        return self.__repr__()

    def add_keywords(self, keywords, sep=None):
        """Adds new keywords x to list of current keywords.

        Args:
            keywords (string, list of strings, techminer.Keywords): new keywords to be added.
            sep (character): separator character in string lists.

        Returns:
            Nothing

        Examples
        ----------------------------------------------------------------------------------------------

        >>> keywords = Keywords()
        >>> keywords = keywords.add_keywords('ann')
        >>> keywords
        [
          "ann"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=False
        >>> keywords = keywords.add_keywords('RNN')
        >>> keywords
        [
          "RNN",
          "ann"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=False
        >>> keywords = keywords.add_keywords(['deep learning', 'fuzzy'])
        >>> keywords
        [
          "RNN",
          "ann",
          "deep learning",
          "fuzzy"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=False
        >>> other_keywords_list = Keywords().add_keywords(['a', 'b', 'c'])
        >>> keywords = keywords.add_keywords(other_keywords_list)
        >>> keywords
        [
          "RNN",
          "a",
          "ann",
          "b",
          "c",
          "deep learning",
          "fuzzy"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=False

        """
        if keywords is None:
            return

        if isinstance(keywords, str):
            keywords = [keywords]

        if isinstance(keywords, Keywords):
            keywords = keywords._keywords

        if isinstance(keywords, pd.Series):
            keywords = keywords.tolist()

        if sep is not None:
            keywords = [
                z.strip()
                for y in keywords
                if y is not None
                for z in y.split(sep)
                if z.strip() != ""
            ]
        else:
            keywords = [
                y.strip() for y in keywords if y is not None and y.strip() != ""
            ]

        if self._keywords is None:
            self._keywords = sorted(list(set(keywords)))
        else:
            keywords.extend(self._keywords)
            self._keywords = sorted(list(set(keywords)))

        self._patterns = None
        return self

    def __len__(self):
        """Returns the number of keywords.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> len(Keywords(['Big data', 'neural networks']))  # doctest: +NORMALIZE_WHITESPACE
        2
        """
        return len(self._keywords)

    def compile(self):
        """Compiles regular expressions.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> x = Keywords(['Big data', 'neural networks'])
        >>> x
        [
          "Big data",
          "neural networks"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=False
        >>> x = x.compile()
        >>> x
        [
          "Big data",
          "neural networks"
        ]
        ignore_case=True, full_match=False, use_re=False, compiled=True

        """
        patterns = self._keywords
        if self._use_re is False:
            patterns = [re.escape(pattern) for pattern in patterns]
        if self._full_match is True:
            patterns = ["^" + pattern + "$" for pattern in patterns]
        if self._ignore_case is True:
            self._patterns = [re.compile(pattern, re.I) for pattern in patterns]
        else:
            self._patterns = [re.compile(pattern) for pattern in patterns]
        return self

    def extract_from_text(self, x, sep=";"):
        r"""Returns a new string with the keywords in string x matching the list of 
        keywords used to fit the model.

        The funcion allows the extraction of complex patterns using regular expresions (regex).
        Detail information about regex sintax in Python can be obtained at 
        https://docs.python.org/3/library/re.html#re-syntax.

        Args:
            x (string): A string object.

        Returns:
            String.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> keywords = Keywords([r"xxx", r"two", r"yyy"])
        >>> keywords = keywords.compile()
        >>> keywords.extract_from_text('one two three four five')
        'two'


        """
        if x is None:
            return None
        result = []
        for pattern in self._patterns:
            match = pattern.search(x)
            if match is not None:
                result.extend([match[0]])
        if len(result):
            return sep.join(sorted(list(set(result))))
        return None

    def __contains__(self, x):
        """Implements in operator.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> x = ['Big data', 'neural networks']
        >>> 'Big data' in Keywords(x).compile()  # doctest: +NORMALIZE_WHITESPACE
        True
        >>> 'big data' in Keywords(x).compile()  # doctest: +NORMALIZE_WHITESPACE
        True
        >>> 'deep learning' in Keywords(x).compile()  # doctest: +NORMALIZE_WHITESPACE
        False
        >>> 'big data' in Keywords(x, ignore_case=False).compile()  # doctest: +NORMALIZE_WHITESPACE
        False

        """
        if self._patterns is None:
            self.compile()
        if self.extract_from_text(x) is None:
            return False
        return True

    def remove_from_text(self, x):
        """Returns a string removing the strings that match a
        list of keywords from x.

        Args:
            x (string): A string object.

        Returns:
            String.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> Keywords('aaa').compile().remove_from_text('1 aaa 2')
        '1  2'

        >>> Keywords('aaa').compile().remove_from_text('1 2')
        '1 2'

        >>> Keywords('aaa').compile().remove_from_text('1 aaa 2 1 2')
        '1  2 1 2'

        >>> Keywords(['aaa', 'bbb']).compile().remove_from_text('1 aaa bbb 2 1 aaa 2')
        '1   2 1  2'

        """
        if x is None:
            return None
        for pattern in self._patterns:
            x = pattern.sub(repl="", string=x)
        return x

    def new_keywords(
        self, x, sep=None,
    ):
        """Creates a new Keywords object by applying the current Keywords to x.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> x = ['11', '111', '11 11 ', 'a', 'b', 'c']
        >>> keywords = Keywords('1.*', use_re=True)
        >>> keywords = keywords.compile()
        >>> keywords.new_keywords(x)
        [
          "11",
          "11 11",
          "111"
        ]
        ignore_case=True, full_match=False, use_re=True, compiled=False


        """
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
        x = [self.extract_from_text(w) for w in x]
        return Keywords(
            keywords=x,
            ignore_case=self._ignore_case,
            full_match=self._full_match,
            use_re=self._use_re,
        )

    # def delete_keyword(self, x):
    #     """Remove string x from the keywords list.
    #     """
    #     self._keywords.remove(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
