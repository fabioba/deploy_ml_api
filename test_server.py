from fastapi.testclient import TestClient
from typing import List

# Import our app from main.py.
from main import app
from pydantic import BaseModel

class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
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

def test_run_inference():
    cens1=Census(39, 'State-gov',77516, 'Bachelors',13, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male',2174,0,40, 'United-States','<=50K')
    cens2=Census(49, 'State-gov',40516, 'Bachelors',80, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Female',2174,0,60, 'United-States','>50K')

    list_inf=list(cens1,cens2)

    r=client.post("/inference/",json=list_inf)
    assert r.status_code == 200

    content=r.json()

    assert content['preds'][0] != None

def test_response_inference():
    cens1=Census(39, 'State-gov',77516, 'Bachelors',13, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male',2174,0,40, 'United-States','<=50K')
    cens2=Census(49, 'State-gov',40516, 'Bachelors',80, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Female',2174,0,60, 'United-States','>50K')

    list_inf=list(cens1,cens2)

    r=client.post("/inference/",json=list_inf)
    assert r.status_code == 200

    content=r.json()

    assert content['preds'][1] !=None