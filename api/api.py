from fastapi import FastAPI
from .model import model as model_lib
app = FastAPI()
import dvc.api
import pandas as pd
import pickle

@app.get("/")
def default():
    return {"output":"Hello World!"}

@app.post("/inference")
def run_inference(item_id: int):
    """
    This method runs inference. Steps:
    1. Get pre-trained model
    2. Read data from file
    3. Inference on data
    """
    # read model
    model_trained = pickle.loads(
    dvc.api.read(
        path='model/model_trained/model.pkl',
        repo='https://github.com/fabioba/deploy_ml_api',
        mode='rb'))

    # read data
    with dvc.api.open(
            path='data/census_clean.csv',
            repo='https://github.com/fabioba/deploy_ml_api') as fd:
            df=pd.read_csv(fd)

    model_lib.inference(model_trained,df)