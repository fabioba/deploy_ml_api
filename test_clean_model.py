import pytest
from .model import model
from pathlib import Path
from pandas.api.types import is_numeric_dtype

def test_numerical_output(data):
    """
        This test checks if salary variable is numeric

        Args:
            df(pandas DF)
    """
    df_preprocess=model.preprocess_step(data)

    assert is_numeric_dtype(df_preprocess['salary']),'salary not numeric'
        



def test_census_exist():
    """
        This test checks if census_clean metadata exists
    """
    my_file = Path(__file__).parent / 'data/census_clean.csv.dvc'

    assert my_file.is_file(),'file does not exist'


def test_model_exist():
    """
        This test checks model metadata exists
    """
    my_file = Path(__file__).parent / 'model/model_trained/model.pkl.dvc'

    assert my_file.is_file(),'model does not exist'    

