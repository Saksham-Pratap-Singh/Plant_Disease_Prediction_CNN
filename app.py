import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# -----------------------------
# Load your trained Keras model
# -----------------------------
MODEL_PATH = 'plant_disease_prediction_model.keras'  # make sure it's in the same folder
model = load_model(MODEL_PATH)

# -----------------------------
# Define prediction function
# -----------------------------
def predict_disease(image):
    image = image.resize((224, 224))  # use your model input size
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if your model was trained that way

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image and let the model predict the disease!")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Predicting...")
    pred = predict_disease(image)

    # Example: change this list to your class labels
    class_names = ['Apple Scab', 'Apple Rust', 'Healthy', 'Potato Early Blight', 'Potato Late Blight']

    st.success(f"✅ Prediction: **{class_names[pred]}**")
