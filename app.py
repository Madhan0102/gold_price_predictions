import streamlit as st
import numpy as np
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    regressor = joblib.load(r'gold_price_model.pkl','readwrite')
    return regressor

regressor = load_model()

# Title of the web app
st.title("Gold Price Prediction")

# User inputs for each feature
spx = st.number_input("S&P 500 Index (SPX)", min_value=0.0, max_value=5000.0, value=3000.0)
gld = st.number_input("Gold ETF (GLD)", min_value=0.0, max_value=500.0, value=120.0)
uso = st.number_input("Oil ETF (USO)", min_value=0.0, max_value=200.0, value=40.0)
slv = st.number_input("Silver ETF (SLV)", min_value=0.0, max_value=100.0, value=15.0)

# Assuming the model was trained on SPX, GLD, USO, and SLV
input_data = np.array([[spx, gld, uso, slv]])

# When the user clicks the "Predict" button
if st.button("Predict"):
    prediction = regressor.predict(input_data)
    
    # Display the prediction result
    st.write(f"The predicted gold price is: ${prediction[0]:.2f}")