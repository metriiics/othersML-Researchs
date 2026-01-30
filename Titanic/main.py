import json
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('titanic_model.pkl')

MODEL_INFO = {
    "name": "Titanic Prediction",
    "author": "Alexsey Kochetkov",
    "version": 1.01,
    "date": "2022-05-31T19:09:57.993322",
    "type": "XGBClassifier",
    "accuracy": 0.829932
}


class Form(BaseModel):
    Pclass: int
    Sex: int	
    Age: float	
    SibSp: int
    Parch: int
    Fare: float
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int

class Prediction(BaseModel):
    Result: int

@app.get("/status")
def status():
    return "I'm ok"

@app.get("/version")
def version():
    return MODEL_INFO

@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model.predict(df)
    
    return {'Result': y}