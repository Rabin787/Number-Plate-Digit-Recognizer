from fastapi import FastAPI, File, UploadFile, Header, HTTPException
import cv2
import numpy as np
import joblib
import tempfile
import os

API_KEY = "rabin-ml-project-2025-secure-key"


app = FastAPI(title="Number Plate Digit Recognition API")

Cmodel = joblib.load("Number_Plate_Digit_Classifier.pkl")


def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.flatten().reshape(1, -1) / 255.0
    return img


@app.post("/predict/model")
async def predict_model(file: UploadFile = File(...), x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        img = preprocess_image(tmp_path)
        prediction = Cmodel.predict(img)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {
        "method": "ml_model",
        "prediction": prediction
    }



from agent import classify_image_gemini

@app.post("/predict/agent")
async def agent_predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    verify_api_key(x_api_key)

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    image_bytes = await file.read()

    try:
        result = classify_image_gemini(
            image_bytes=image_bytes,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
