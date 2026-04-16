import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Prediction App")

st.write("Enter flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# Prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Flower: {species[prediction[0]]}")
