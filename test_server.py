from fastapi.testclient import TestClient

# Import our app from main.py.
import server

# Instantiate the testing client with our app.
client = TestClient(server)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.content!=None

def test_run_inference():
    body={"num_rows":10}
    r=client.post("/inference",json=body)
    content=r.json()

    assert content['preds'] != None

def test_response_inference():
    body={"num_rows":10}
    r=client.post("/inference",json=body)
    content=r.json()

    assert len(content['preds']) == 20