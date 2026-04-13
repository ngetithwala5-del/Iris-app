import joblib
model = joblib.load("digit_model.pkl")

from PIL import Image
import numpy as np
import streamlit as st

def preprocess_image(image):
    image = image.convert('L')  # grayscale
    image = image.resize((8, 8), Image.Resampling.LANCZOS)

    image_array = np.array(image)

    # 🔥 Invert colors (important!)
    image_array = 255 - image_array  

    image_array = np.where(image_array > 128, 255, 0)

    # scale to 0–16
    image_scaled = (16 * (image_array / 255)).astype(np.float32)

    return image_scaled.flatten().reshape(1, -1)

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=150)

    processed_image = preprocess_image(image)

    # 👇 Add button
    if st.button("Predict"):
        prediction = model.predict(processed_image)
        st.success(f"Prediction: {prediction[0]}")
