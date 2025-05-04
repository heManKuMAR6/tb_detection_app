# tb_detection_app/app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from utils.preprocess import prepare_image
from utils.gradcam import generate_gradcam

# Load the trained model
model = load_model("model/tb_cnn_model.h5")

# Streamlit UI
st.set_page_config(page_title="TB Detection", layout="centered")
st.title("\U0001F4CB Tuberculosis Detection from Chest X-rays")
st.markdown("Upload a chest X-ray image to predict TB and visualize Grad-CAM heatmap.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = prepare_image(image)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "TB Positive" if prediction > 0.5 else "TB Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: **{label}**")
    st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

    # Grad-CAM Heatmap
    heatmap_img = generate_gradcam(model, img_array, last_conv_layer_name="conv5_block16_2_conv")
    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_column_width=True)