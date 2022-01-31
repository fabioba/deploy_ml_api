"""
This module has been created to read df and clean data

Author: Fabio
Date: 29th Jan, 2022
"""

import pandas as pd
import logging

FORMAT = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_column_name(df):
    """
        This method gets a df as input an remove spaces from their columns names.
        Replace dash with underscore

        Args:
            df(Pandas DF)
    """
    try:
        logger.info('START')

        df.columns = [item.replace(' ', '') for item in df.columns]
        df.columns = [item.replace('-', '_') for item in df.columns]

        logger.info('SUCCESS')
    except Exception as err:
        logger.error(err)


def remove_space_value(df):
    """
        This method gets a df as input an remove spaces from their values

        Args:
            df(Pandas DF)
    """
    try:
        logger.info('START')

        list_cols=['education','marital_status','native_country',
                    'occupation','race','relationship','sex','workclass','salary']

        for column_name in list_cols:
            df[column_name] = df[column_name].map(lambda x: x.replace(' ',''))

        logger.info('SUCCESS')
    except Exception as err:
        logger.error(err)


def store_cleaned_df(df):
    """
        This method stores cleaned df

        Args:
            df(Pandas DF)
    """
    try:
        logger.info('START')

        df.to_csv('./data/census_clean.csv', index=False)

        logger.info('SUCCESS')
    except Exception as err:
        logger.error(err)


if __name__ == '__main__':

    logger.info('MAIN')

    df = pd.read_csv('./data/census.csv')

    # remove spaces from columns name
    prepare_column_name(df)

    remove_space_value(df)

    # store updated df
    store_cleaned_df(df)
