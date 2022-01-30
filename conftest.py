import pandas as pd
from pathlib import Path
import pytest
import dvc

@pytest.fixture(scope='session')
def data():
    """
        This method returns the census clean as data

        Output:
            df(pandas DF)
    """

    with dvc.api.open(
            path='data/census_clean.csv',
            repo='https://github.com/fabioba/deploy_ml_api') as fd:
            df=pd.read_csv(fd)

    print('read data from dvc')
    return df