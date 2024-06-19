from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
import os

load_dotenv()

app = FastAPI()

# Global variable to hold the loaded model
loaded_model = None

# Function to load the TensorFlow.js model
def load_tfjs_model(model_path):
    global loaded_model
    try:
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, safe_mode=True)
        loaded_model.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
        print("TensorFlow.js model loaded successfully.")
    except Exception as e:
        print(f"Error loading TensorFlow.js model: {e}")

# Load the model during startup
load_tfjs_model(os.getenv("MODEL_PATH"))

@app.get("/")
def config():
    global loaded_model
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")
    return {"configuration": "test"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    global loaded_model
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")

    try:
        image_bytes = await image.read()
        result = await predict_classification(loaded_model, image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

async def predict_classification(model, image):
        image = Image.open(BytesIO(image))
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet.preprocess_input(image)

        prediction = model.predict(image)

        return classification(prediction.tolist()[0])

def classification(result):
    acne = result[0]
    redness = result[1]
    eyebags = result[2]

    if (acne > redness and acne > eyebags):
        return {"issue": "Acne", "score": acne}
    elif (redness > acne and redness > eyebags):
        return {"issue": "Redness", "score": redness}
    elif (eyebags > acne and eyebags > redness):
        return {"issue": "Eyebags", "score": redness}