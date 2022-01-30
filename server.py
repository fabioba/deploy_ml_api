from fastapi import FastAPI
from model import model as model_lib
import dvc.api
import pandas as pd
import pickle
from pydantic import BaseModel

class Inference(BaseModel):
    num_rows: int

app = FastAPI()




@app.get("/")
def default():
    return {"output":"Hello World!"}

@app.post("/inference/")
async def inference(inf_obj: Inference):
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

    # read data
    with dvc.api.open(
            path='data/census_clean.csv',
            repo='https://github.com/fabioba/deploy_ml_api') as fd:
            df=pd.read_csv(fd,nrows=inf_obj.num_rows)
    # run inference
    preds=model_lib.inference(model_trained,df)

    return {"preds": f"{preds}, num_rows: {inf_obj.num_rows}"}