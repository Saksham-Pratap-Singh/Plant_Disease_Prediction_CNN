import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import os

import gdown

MODEL_URL = "https://drive.google.com/uc?id=1w-m6L3Z6M0QNBENbaRLcOQ8oAJRaANJB"
MODEL_PATH = "plant_disease_prediction_model.h5"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("✅ Model downloaded!")


# --- Streamlit UI ---
st.title("🌿 Plant Disease Prediction App")
st.write("Upload a plant leaf image to predict the disease")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128, 128))  # adjust size if different in training
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Prediction
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)

    st.success(f"✅ Prediction class: {pred_class[0]}")
