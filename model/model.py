"""
This module contains model training and pre-processing steps.

Author: Fabio
Date: 29th of Jan, 2022
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
import pickle 
import os

FORMAT = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_step(df):
    """
    Preprocess steps includes getting dummies variables from categorical and split df into train and test set.

    Args:
        df(Pandas df)

    Output:
        x_train(array)
        X_test(array)
        y_train(array)
        y_test(array)
    """
    try:
        logger.info('START')

        # create dummies vars
        df_preprocess = pd.get_dummies(df, columns =['capital-loss', 'education', 'relationship', 'age', 'native-country', 'workclass', 'capital-gain', 'marital-status', 'hours-per-week', 'fnlgt', 'education-num', 'occupation', 'sex', 'race'])
        
        # create X and Y
        X=df_preprocess.drop(['salary'],axis=1).values
        df_preprocess.loc[df_preprocess['salary']=='<=50K','salary']=0
        df_preprocess.loc[df_preprocess['salary']=='>50K','salary']=1
        y=df_preprocess['salary'].values

        # convert Y type
        y=y.astype('int')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        logger.info('SUCCESS')


        return X_train, X_test, y_train, y_test 

    except Exception as err:
        logger.error(err)

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    try:
        logger.info('START')

        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)

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

        path_file=os.path.join(model_folder,model_name)

        with open(path_file,'wb') as f:
            pickle.dump(model,f)

        logger.info('SUCCESS')

    except Exception as err:
        logger.error(err)

if __name__=='__main__':
    logger.info('MAIN')

    df=pd.read_csv('./data/census_clean.csv')

    X_train, X_test, y_train, y_test =preprocess_step(df)
    
    model=train_model(X_train,y_train)

    store_model(model,'model.pkl','./model_trained')



    