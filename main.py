# Put the code for your API here.
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
import pickle
import pandas as pd
from typing import List
from starter.ml.data import process_data
from starter.train_model import cat_features
from starter.ml.model import compute_model_metrics

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

# @app.get("/")
# async def say_hello():
#     return {"greeting": "Hello World!"}


class exmaple(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str


@app.post("/predict/")
async def predict(test_sample: List[exmaple]):
    # if I should read from s3 bucket
    model = pickle.load(open("./model/lr_model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    lb = pickle.load(open("./model/lb.pkl", "rb"))
    test_example = pd.DataFrame([jsonable_encoder(ex) for ex in test_sample])
    X_test, y_test, _, _ = process_data(
        test_example,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False,
    )
    y_pred = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    return {"precision": precision, "recall": recall, "fbeta": fbeta}


@app.get("/info/")
def welcome():
    return {
        "welcome": "Here is the API where you can get predictions for your salary next year"
    }
