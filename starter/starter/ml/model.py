import sys
sys.path.append("/Users/jielyu/udacity/mle/nd0821-c3-starter-code/starter")
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from starter.ml.data import process_data_with_one_fixed_feature
import pickle
import numpy as np
from starter.train_model import cat_features

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

    


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def partial_inference(model, X: pd.DataFrame, fixed_feature:str):
    model = pickle.load(open("./lr_model.pkl", "rb"))
    X_test, y_test = process_data_with_one_fixed_feature(X, categorical_features=cat_features, label="salary", fixed_feature=fixed_feature)
    unique_values = np.unique(X[:,-1])
    precisions, recalls, fbeta_scores = [], [], []
    for value in unique_values:
        X_test = X_test[X_test[:,-1]==value]
        y_pred = inference(model, X_test)
        p,r,f = compute_model_metrics(y_test, y_pred)
        precisions.append(p)
        recalls.append(r)
        fbeta_scores.append(f)
    
    return precisions, recalls, fbeta_scores


    

    