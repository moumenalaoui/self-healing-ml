import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load trained model
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

# Define expected input structure
class InputData(BaseModel):
    age: int
    workclass: int
    fnlwgt: int
    education: int
    education_num: int
    marital_status: int
    occupation: int
    relationship: int
    race: int
    sex: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: int

app = FastAPI()

@app.get("/")
def root():
    return {"message": "ML model API is running."}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
