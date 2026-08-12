from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model("model.keras")
IMG_SIZE = (256, 256)
@app.get("/")
def home():
    return {"message": "Model API is running. Go to /docs to try it out."}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_array)
    score = float(prediction[0][0])
    return {
        "score": score,
        "class": "class_1" if score > 0.5 else "class_0"
    }
    
@app.get("/health")
def health():
    return {"status": "ok"}
