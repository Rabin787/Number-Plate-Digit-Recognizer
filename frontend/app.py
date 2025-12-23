import streamlit as st
import requests
from PIL import Image

API_URL_MODEL = "http://localhost:8000/predict/model"
API_URL_AGENT = "http://localhost:8000/predict/agent"
API_KEY = "rabin-ml-project-2025-secure-key"

st.set_page_config(page_title="Nepali Number Plate Recognition")

st.title("🚗 Nepali Number Plate Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Nepali Number Plate Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

method = st.radio(
    "Choose Prediction Method",
    ["ML Model", "Agent"]
)

if uploaded_file and st.button("Predict"):
    headers = {"x-api-key": API_KEY}

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    with st.spinner("Predicting..."):
        if method == "ML Model":
            response = requests.post(
                API_URL_MODEL,
                files=files,
                headers=headers
            )
        else:
            response = requests.post(
                API_URL_AGENT,
                files=files,
                headers=headers
            )


    if response.status_code == 200:
        st.success("Prediction Successful")

        result = response.json()

        if method == "ML Model":
            st.write("### 🔢 Predicted Digit")
            st.write(result["prediction"])
        else:
            st.write("### 🤖 Agent Decision")
            st.json(result)

    else:
        st.error("Prediction failed")
        st.code(response.text)
