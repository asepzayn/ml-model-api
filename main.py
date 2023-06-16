import os
import uvicorn
import traceback
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import decode_batch_predictions, preprocess_audio

model = tf.keras.models.load_model('h5_model1.h5', compile=False)

app = FastAPI()

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

class ResponseText(BaseModel):
    transcription: str

@app.post("/predict")
def predict(response: Response, file: UploadFile = UploadFile(...)):
    try:
        preprocessed_audio = preprocess_audio(file.file.read())
        # Reshape the audio data for model input
        input_data = tf.expand_dims(preprocessed_audio, axis=0)
        # Make the prediction using the loaded model
        prediction = model.predict(input_data)
        predicted_text = decode_batch_predictions(prediction)[0]
        # Return the predicted text as the API response
        return {"predicted_text": predicted_text}
    
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"
    
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)
