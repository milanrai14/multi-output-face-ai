import streamlit as st
import requests
from PIL import Image

st.title("FaceIQ - Age & Gender Predictor")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction done!")
                st.write("**Age:**", result["age"])
                st.write("**Gender:**", result["gender"])
                st.write("**Gender Probability:**", round(result["gender_probability"], 3))
            else:
                st.error("Something went wrong. Check if FastAPI server is running.")
