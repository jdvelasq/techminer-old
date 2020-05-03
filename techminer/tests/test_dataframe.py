import pytest

import pandas as pd
import pandas._testing as tm

# from dataframe import DataFrame
from techminer import DataFrame


@pytest.fixture
def testdf():
    return pd.DataFrame(
        {
            "Authors": "author 0,author 1;author 2;author 2,author 3;author 3;author 4,author 5;author 5,author 5".split(
                ";"
            ),
            "Author (s) ID": "id0;id1,id2,id2;id3,id3,id4;id5,id6;id7".split(","),
            "ID": list(range(6)),
            "Cited by": list(range(6)),
        }
    )


def test_explode():

    testdf = pd.DataFrame(
        {
            "Authors": "author 0,author 1,author 2;author 3;author 4".split(";"),
            "ID": list(range(3)),
        }
    )

    expected = pd.DataFrame(
        {
            "Authors": "author 0;author 1;author 2;author 3;author 4".split(";"),
            "ID": [0, 0, 0, 1, 2],
        }
    )

    result = DataFrame(testdf).explode("Authors")
    tm.assert_frame_equal(result, expected)
