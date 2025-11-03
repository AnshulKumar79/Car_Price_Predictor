# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 23:28:00 2025

@author: Lenovo
"""

import streamlit as st
import pickle
import numpy as np

# --- 1. Define the mappings ---
# These must match the mappings you used during training
fuel_map = {'Diesel': 0, 'Petrol': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
seller_map = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
transmission_map = {'Manual': 0, 'Automatic': 1}
owner_map = {
    'Test Drive Car': 0, 
    'First Owner': 1, 
    'Second Owner': 2, 
    'Third Owner': 3, 
    'Fourth & Above Owner': 4
}

# --- 2. Load the pickled model ---
model_path = 'lin_reg_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found. Please ensure '{model_path}' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()

# --- 3. Set up the Streamlit app interface ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction")
st.markdown("This app predicts the price of a car based on its features.")

# --- 4. Create columns for inputs ---
col1, col2 = st.columns(2)

with col1:
    st.header("Numerical Features")
    year = st.number_input("Year", min_value=1990, max_value=2024, value=2015, help="Enter the manufacturing year.")
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, help="Enter the total kilometers driven.")

with col2:
    st.header("Categorical Features")
    # Use st.selectbox to create dropdowns
    fuel_selection = st.selectbox(
        "Fuel Type", 
        options=list(fuel_map.keys())
    )
    
    seller_selection = st.selectbox(
        "Seller Type", 
        options=list(seller_map.keys())
    )
    
    transmission_selection = st.selectbox(
        "Transmission", 
        options=list(transmission_map.keys())
    )
    
    owner_selection = st.selectbox(
        "Owner", 
        options=list(owner_map.keys())
    )

# --- 5. Prediction logic ---
st.divider()
if st.button("Predict Car Price", use_container_width=True, type="primary"):
    # Convert selections to their encoded values
    try:
        fuel = fuel_map[fuel_selection]
        seller_type = seller_map[seller_selection]
        transmission = transmission_map[transmission_selection]
        owner = owner_map[owner_selection]
        
        # Create the feature array in the correct order
        # ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
        features = np.array([[
            year, 
            km_driven, 
            fuel, 
            seller_type, 
            transmission, 
            owner
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display the prediction
        st.success(f"**Predicted Car Price: ₹ {prediction[0]:,.2f}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("This tool is for educational purposes. The model's accuracy depends on the data it was trained on.")

