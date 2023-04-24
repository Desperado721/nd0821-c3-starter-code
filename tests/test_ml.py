import pytest
import pandas as pd
from starter.ml.model import compute_model_metrics,inference
from starter.ml.data import process_data
from starter.train_model import cat_features
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pickle


@pytest.fixture
def raw_data():
    data = pd.read_csv("../../../data/census.csv")
    return data

@pytest.fixture
def lr():
    lr = pickle.load("../../../model/lr_model.pkl")
    return lr


def test_process_data(data):
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert type(X_train)==np.array
    assert type(y_train) == np.array
    assert np.unique(y_train)[0] == 0
    assert np.unique(y_train)[1] ==1
    assert type(encoder) == OneHotEncoder
    assert type(lb) == LabelBinarizer


def test_inference(lr, data):
    X_train, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    preds = inference(lr, X_train)
    assert type(preds) == np.array


    

def test_compute_metrics(lr, data):
    X_train, y_train, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )    

    y_preds = inference(lr, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, y_preds)
    assert precision >= 0 and precision <=1
    assert recall >=0 and recall <=1
    assert fbeta >=0 and fbeta <=1

