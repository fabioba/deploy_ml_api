from fastapi.testclient import TestClient

# Import our app from main.py.
import server

# Instantiate the testing client with our app.
client = TestClient(server)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_run_inference():
    r=client.get("/inference/10")
    content=r.json()

    assert content['preds'] != None
    assert len(content['preds']) == 10

def test_malformed_url():
    r=client.get("/inference/hello")
    
    assert r!=200
