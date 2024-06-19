from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

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
load_tfjs_model('./model/my_model.h5')

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
        score = await tf.make_ndarray(prediction)
        confidence_score = np.max(score) * 100

        return {'confidenceScore': confidence_score}