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


def preprocess_step(df):
    """
    Preprocess steps includes getting dummies variables from categorical and split df into train and test set.

    Args:
        df(Pandas df)

    Output:
        x_train(array)
        y_train(array)
        x_test(array)
        y_test(array)
    """
    try:
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


        return X_train, X_test, y_train, y_test 



    except Exception as err:
        logging.error(err)

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

    pass


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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


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
    pass