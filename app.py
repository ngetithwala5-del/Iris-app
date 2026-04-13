import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Prediction App")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict button
if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")
