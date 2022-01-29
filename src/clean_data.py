"""
This module has been created to read df and clean data

Author: Fabio
Date: 29th Jan, 2022
"""

import pandas as pd
import logging

logger=logging.getLogger()
 

def prepare_column_name(df):
    """
        This method gets a df as input an remove spaces from their columns names

        Args:
            df(Pandas DF)
    """
    try:
        logger.info('START')

        df.columns=[item.replace(' ','') for item in df.columns]


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

        df.to_csv('./data/census_clean.csv', index = False)


        logger.info('SUCCESS')
    except Exception as err:
        logger.error(err)   


if __name__=='__main__':

    df=pd.read_csv('./data/census.csv')

    # remove spaces from columns name
    prepare_column_name(df)

    # store updated df
    store_cleaned_df(df)



