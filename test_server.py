from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

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
    body={"num_rows":10}
    r=client.post("/inference/",json=body)
    assert r.status_code == 200

    content=r.json()

    assert content['preds'] != None

def test_response_inference():
    body={"num_rows":10}
    r=client.post("/inference/",json=body)
    assert r.status_code == 200

    content=r.json()

    assert content['num_rows'] == 10