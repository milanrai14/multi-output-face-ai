from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from prediction import predict_age_gender
from schema import PredictionResponse
from model_loader import load_model

app = FastAPI(title="FaceIQ - Age & Gender Prediction")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return {"message": "FaceIQ API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = predict_age_gender(image)

    return result
