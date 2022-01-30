import pandas as pd
from pathlib import Path
import pytest

@pytest.fixture(scope='session')
def data():
    """
        This method returns the census clean as data

        Output:
            df(pandas DF)
    """
    df=pd.read_csv(str(Path(__file__).parent / 'data/census_clean.csv'))

    print('read data')
    return df