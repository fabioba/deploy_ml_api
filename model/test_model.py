import pytest
import model
from pandas.api.types import is_numeric_dtype

def test_numerical_output(df,output):
    """
        This test checks if the output variable is numeric

        Args:
            df(pandas DF)
            output(str): name of the otput column
    """
    try:
        df_preprocess=model.preprocess_step(df)

        assert is_numeric_dtype(df_preprocess[output]),'output not numeric'
    except AssertionError as err:
        print('ASSERT:{}'.format(err))
    except Exception as err:
        print(err)
