from fastapi.testclient import TestClient
from typing import List
import numpy as np
# Import our app from main.py.
from main import app
from pydantic import BaseModel

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


# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")

    assert r.status_code == 200

    content=r.json()
    assert content!=None
    assert content['output']=='Hello World!'

def test_run_inference_negative():
    cens1={"age":39, "workclass":'State-gov',
    "fnlgt":77516, "education":'Bachelors',"education_num":8, "marital_status":'Never-married', "occupation":'Adm-clerical',"relationship":'Not-in-family', "race":'White', "sex":'Male',"capital_gain":2174,"capital_loss":0,"hours_per_week":40, "native_country":'United-States',"salary":'<=50K'}
    
    r=client.post("/inference/",json=cens1)
    assert r.status_code == 200

    content=r.json()

    assert content['preds']['0']==0

def test_run_inference_positive():
    cens1={"age":31, "workclass":'Private',
    "fnlgt":292516, "education":'Masters',"education_num":14, "marital_status":'Divorced', "occupation":'Exec-managerial',"relationship":'Unmarried', "race":'White', "sex":'Female',"capital_gain":14084,"capital_loss":0,"hours_per_week":60, "native_country":'United-States',"salary":'>50K'}

    r=client.post("/inference/",json=cens1)
    assert r.status_code == 200

    content=r.json()

    assert content['preds']['0'] ==0