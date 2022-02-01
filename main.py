from fastapi import FastAPI
from model import model as model_lib
import dvc.api
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
from typing import List
from fastapi.encoders import jsonable_encoder
import json

class Census(BaseModel):
    age:  int
    workclass: str
    fnlgt: int
    education: str
    education_num:  int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain:  int
    capital_loss:  int
    hours_per_week:  int
    native_country: str
    salary: str

app = FastAPI()




@app.get("/")
def default():
    return {"output":"Hello World!"}

@app.post("/inference/")
async def inference(list_inf_obj: Census):
    """
    This method runs inference on a specific number of rows.
    
    Steps:
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

    # convert input
    item=list_inf_obj.dict()

    df=pd.DataFrame(item, index=[0])

    df_preprocessed=model_lib.preprocess_step_s3(df)

    # run inference
    preds=model_lib.inference(model_trained,df_preprocessed)

    return {"preds": preds}