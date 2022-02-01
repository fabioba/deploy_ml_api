"""
This module contains model training and pre-processing steps.

Author: Fabio
Date: 29th of Jan, 2022
"""
from importlib.machinery import DEBUG_BYTECODE_SUFFIXES
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
import pickle
from sklearn import preprocessing
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import random
import sys
import joblib
from io import BytesIO
import dvc.api

sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import dump, load


FORMAT = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_step_s3(df):
    """
    Preprocess steps includes getting dummies variables from categorical and split df into train and test set.

    Args:
        df(Pandas df)

    Output:
        df(Pandas df)
    """
    try:

        df_preprocess=df.copy()
        logger.info('START')

        # create dummies vars for categoricals
        #df_preprocess = pd.get_dummies(df_preprocess, columns =['education','marital_status','native_country','occupation','race','relationship','sex','workclass'])
        
        # standardize numericals
        sc=load(BytesIO(
            dvc.api.read(
                    path='model/model_trained/std_scaler.bin',
                    repo='https://github.com/fabioba/deploy_ml_api',
                    mode='rb')))
        df_preprocess[['capital_loss', 'age', 'hours_per_week', 'fnlgt', 'education_num', 'capital_gain']]=sc.fit_transform(df_preprocess[['capital_loss', 'age', 'hours_per_week', 'fnlgt', 'education_num', 'capital_gain']].values)
                
        # convert categorical output into numerical
        df_preprocess.loc[df_preprocess['salary']=='<=50K','salary']=0
        df_preprocess.loc[df_preprocess['salary']=='>50K','salary']=1
        df_preprocess['salary']=df_preprocess['salary'].astype('int') 

        logger.info('SUCCESS')

        return df_preprocess

    except Exception as err:
        logger.error(err)
def preprocess_step(df):
    """
    Preprocess steps includes getting dummies variables from categorical and split df into train and test set.

    Args:
        df(Pandas df)

    Output:
        df(Pandas df)
    """
    try:

        df_preprocess=df.copy()
        logger.info('START')

        # create dummies vars for categoricals
        #df_preprocess = pd.get_dummies(df_preprocess, columns =['education','marital_status','native_country','occupation','race','relationship','sex','workclass'])
        
        # standardize numericals
        sc=preprocessing.StandardScaler()
        df_preprocess[['capital_loss', 'age', 'hours_per_week', 'fnlgt', 'education_num', 'capital_gain']]=sc.fit_transform(df_preprocess[['capital_loss', 'age', 'hours_per_week', 'fnlgt', 'education_num', 'capital_gain']].values)
        
        # store standard scaler
        dump(sc, str(Path(__file__).parent / 'model_trained/std_scaler.bin'), compress=True)
        
        # convert categorical output into numerical
        df_preprocess.loc[df_preprocess['salary']=='<=50K','salary']=0
        df_preprocess.loc[df_preprocess['salary']=='>50K','salary']=1
        df_preprocess['salary']=df_preprocess['salary'].astype('int') 

        logger.info('SUCCESS')

        return df_preprocess

    except Exception as err:
        logger.error(err)


def split_ds(df):
    """
    This method splits df

    Args:
        df(Pandas df)

    Output:
        x_train(array)
        X_test(array)
        y_train(array)
        y_test(array)
    """
    try:

        df_preprocess=df.copy()
        logger.info('START')

        # get size of the class with the lowest records
        df_group=df_preprocess.groupby(['salary']).size().reset_index().rename(columns={0:'n_records'})
        min_size=min(df_group['n_records'].values)

        # create class of df. resampling on that with more records
        df_salary_positive=df_preprocess[df_preprocess['salary']==1].reset_index(drop=True)
        df_salary_negative=df_preprocess[df_preprocess['salary']==0].sample(n=min_size).reset_index(drop=True)
        # concat dfs
        df_salary_balanced=df_salary_positive.append(df_salary_negative)

        # get number of sample per class
        first_sample_size=math.floor((len(df_salary_balanced)*0.80)/2)
        # calculate index negative class for train
        train_sample_negative_index=random.sample(list(df_salary_negative.index),first_sample_size)
        # calculate index negative class for test
        test_sample_negative_index=set(list(df_salary_negative.index))-set(train_sample_negative_index)
        # negative train
        df_train_negative=df_salary_negative[df_salary_negative.index.isin(train_sample_negative_index)]
        # negative test
        df_test_negative=df_salary_negative[df_salary_negative.index.isin(test_sample_negative_index)]

        # calculate index positive class for train
        train_sample_positive_index=random.sample(list(df_salary_positive.index),first_sample_size)
        # calculate index positive class for test
        test_sample_positive_index=set(list(df_salary_positive.index))-set(train_sample_positive_index)
        # positive train
        df_train_positive=df_salary_positive[df_salary_positive.index.isin(train_sample_positive_index)]
        # positive test
        df_test_positive=df_salary_positive[df_salary_positive.index.isin(test_sample_positive_index)]

        # total df
        df_train=df_train_negative.append(df_train_positive).reset_index(drop=True)
        df_test=df_test_negative.append(df_test_positive).reset_index(drop=True)

        print(df_train.shape)
        print(df_train.groupby(['salary']).size())
        print(df_test.shape)
        print(df_test.groupby(['salary']).size())



        logger.info('SUCCESS')

        return df_train, df_test

    except Exception as err:
        logger.error(err)

# Optional: implement hyperparameter tuning.
def train_model(df_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : pandas df
        Training data.
    Returns
    -------
    model
        Trained machine learning model.
    """

    try:
        logger.info('START')

        #classifier = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=1000)
        
        #classifier.fit(X_train, y_train)
        classifier = smf.glm('salary ~  capital_loss + age + hours_per_week + fnlgt + education_num + capital_gain + C(education) + C(marital_status) + C(native_country) + C(occupation) + C(race) + C(relationship) + C(sex) + C(workclass)', family=sm.families.Binomial(), data=df_train).fit()

        logger.info('SUCCESS')

        return classifier

    except Exception as err:
        logger.error(err)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    try:
        logger.info('START')

        fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
        precision = precision_score(y, preds, zero_division=1)
        recall = recall_score(y, preds, zero_division=1)

        logger.info('SUCCESS')

        return precision, recall, fbeta

    except Exception as err:
        logger.error(err)


def data_slices_metrics(df,model):
    """
        This method calculate the metrics on each data slice (categorical variable)

        Args:
            df(pandas DF)
            model(pkl)
    """
    try:

        logger.info('START')
        
        data_slice_list=['education','marital_status','native_country','occupation','race','relationship','sex','workclass']

        array_items=[]
        for col in data_slice_list:
            logger.info('col: {}'.format(col))

            list_unique=df[col].unique()

            for item in list_unique:
                logger.info('item: {}'.format(item))

                df_temp=df[df[col]==item]

                df_temp_preprocessed=preprocess_step(df_temp)


                preds=inference(model,df_temp_preprocessed)

                y=df_temp_preprocessed.salary.values


                precision, recall, fbeta=compute_model_metrics(y,preds)

                logger.info(f'precision: {precision}, recall: {recall}, fbeta: {fbeta}')

                data_temp=[[col,item,precision,recall,fbeta]]
                array_items=array_items+data_temp

        df_final=pd.DataFrame(data=array_items,columns=[['col','item','precision','recall','fbeta']])

        df_final.to_csv(str(Path(__file__).parent / 'output/data_slice_metrics.csv'))


        logger.info('SUCCESS')

    except Exception as err:
        logger.error(err)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    try:
        logger.info('START')

        pred=model.predict(X)

        logger.info('SUCCESS')

        return pred
    except Exception as err:
        logger.error(err)

def store_model(model,model_name,model_folder):
    """
        This method stores pre-trained model

        model(sklearn)
        model_name(str)
        model_folder(str)
    """
    try:
        logger.info('START')

        path_file=str(Path(__file__).parent / model_folder / model_name)

        with open(path_file,'wb') as f:
            pickle.dump(model,f)

        logger.info('SUCCESS')

    except Exception as err:
        logger.error(err)

def store_pre_processed_data(df):
    """
        This method stores pre-processed data

        model(df)
    """
    try:
        logger.info('START')

        path_file=str(Path(__file__).parent.parent / 'data/pre_processed_data.csv')

        df.to_csv(path_file)

    except Exception as err:
        logger.error(err)


if __name__=='__main__':
    logger.info('MAIN')

    df=pd.read_csv('./data/census_clean.csv')

    df_preprocessed =preprocess_step(df)

    # store pre-processed data
    store_pre_processed_data(df)
    
    df_train,df_test=split_ds(df_preprocessed)

    model=train_model(df_train)

    store_model(model,'model.pkl','model_trained')

    data_slices_metrics(df_test,model)

    preds_test=inference(model,df_test)
    precision, recall, fbeta=compute_model_metrics(df_test['salary'].values,preds=preds_test)
    logger.info('{} - {} - {}'.format(precision, recall, fbeta))



    