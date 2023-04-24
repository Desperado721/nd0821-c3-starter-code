import pytest
import sys
sys.path.append("/Users/jielyu/udacity/mle/nd0821-c3-starter-code/starter/")
import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# def test_post():
    # data = json.dumps({"value": 10})
    # r = client.post("/42?query=5", data=data)
    # print(r.json())
    # assert r.json()["path"] == 42
    # assert r.json()["query"] == 5
    # assert r.json()["body"] == {"value": 10}

@pytest.fixture
def test_point():    
    return [{
        "age": 0,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": "<=50K"
    }]

@pytest.fixture
def test_points():   
    return [test_point[0]]*2

def test_get_info():
    response = client.get('/info')
    assert response.status_code ==200
    assert response.json() == {"welcome": "Here is the API where you can get predictions for your salary next year"}


def test_post_one_example(test_point):
    test_sample = json.dumps(test_point)
    response = client.post("/predict", data=test_sample)
    assert response.status_code == 200
    assert response.json()['precision'] == 0
    assert response.json()['recall'] == 1
    assert response.json()['fbeta'] == 0

def test_post_two_example(test_points):
    test_samples = json.dumps(test_points)
    response = client.post("/predict", data=test_samples)
    assert response.status_code == 200
    assert response.json()['precision'] == 0

