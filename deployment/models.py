'''
===============================================

Faris Arief Mawardi

Dataset: perth_houses.csv

Dataset Source: https://www.kaggle.com/datasets/syuzai/perth-house-prices

Objective: Creating a Model deployment for the project that has been created

'''

import numpy as np
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import json

def run():
    # Model Loading
    with open('model.pkl', 'rb') as file_1:
        model = pickle.load(file_1)
        st.markdown("<h1 style='text-align: center;'>Welcome to The Prediction Model Page!</h1>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;font-style: italic;'>Please provide all the necessary details and information about the property for price prediction.</h1>", unsafe_allow_html=True)    
        ADDRESS = st.text_input(label='Enter the address of the property you want to predict')
        SUBURB = st.text_input(label='Enter the Suburb/Area of the property you want to predict')
        BEDROOMS = st.number_input(label='Enter the number of bedrooms in the property you want to predict', min_value=1)
        BATHROOMS = st.number_input(label='Enter the number of bathrooms in the property you want to predict', min_value=1)
        GARAGE = st.number_input(label='Enter the maximum capacity of vehicles in the garage of the property you want to predict', min_value=0)
        LAND_AREA = st.number_input(label='Enter the land area of the property you want to predict (in square meters)', min_value=1)
        FLOOR_AREA = st.number_input(label='Enter the floor area of the property you want to predict (in square meters)', min_value=1)
        BUILD_YEAR = st.number_input(label='Enter the year the property was built that you want to predict', min_value=1)
        CBD_DIST = st.number_input(label='Enter the distance from the property to the city center (in meters)', min_value=0)
        NEAREST_STN = st.text_input(label='Enter the name of the Nearest Train Station from your property')
        NEAREST_STN_DIST = st.number_input(label='Enter the distance from your property to the nearest train station (in meters)', min_value=0)
        DATE_SOLD = st.number_input(label='Enter the Last Sale Year of your property', min_value=0)
        POSTCODE = st.number_input(label='Enter the postcode of your property address', min_value=0)
        LATITUDE = st.number_input(label='Enter the latitude of your property location')
        LONGITUDE = st.number_input(label='Enter the longitude of your property location')
        NEAREST_SCH = st.text_input(label='Enter the name of the Nearest School from your property')
        NEAREST_SCH_DIST = st.number_input(label='Enter the distance from your property to the nearest school (in meters)', min_value=0)
        NEAREST_SCH_RANK = st.number_input(label='Enter the rank of the nearest school from your property', min_value=0)

    
    st.write('Below is the result of the data you have input: ')
    
    data_inf = pd.DataFrame({
        "ADDRESS": ADDRESS,
        "SUBURB": SUBURB,
        "BEDROOMS": BEDROOMS,
        "BATHROOMS": BATHROOMS,
        "GARAGE": GARAGE,
        "LAND_AREA": LAND_AREA,
        "FLOOR_AREA": FLOOR_AREA,
        "BUILD_YEAR": BUILD_YEAR,
        "CBD_DIST": CBD_DIST,
        "NEAREST_STN": NEAREST_STN,
        "NEAREST_STN_DIST": NEAREST_STN_DIST,
        "DATE_SOLD": DATE_SOLD,
        "POSTCODE": POSTCODE,
        "LATITUDE": LATITUDE,
        "LONGITUDE": LONGITUDE,
        "NEAREST_SCH": NEAREST_SCH,
        "NEAREST_SCH_DIST": NEAREST_SCH_DIST,
        "NEAREST_SCH_RANK": NEAREST_SCH_RANK
    }, index=[0])


    st.table(data_inf)
    
    if st.button(label='Predict'):
    
        # Perform prediction on the input data
        y_pred_inf = model.predict(data_inf)

        # # Ensure y_pred_inf is a scalar value for formatting purposes
        # predicted_price = y_pred_inf[0] if isinstance(y_pred_inf, (list, pd.Series, pd.DataFrame)) else y_pred_inf
        
        # st.metric(label="Predicted Property Price is ", value=f"${predicted_price[0]:,.2f}" if isinstance(predicted_price, np.ndarray) else f"${predicted_price:,.2f}")
        st.metric(label="Predicted Property Price is ", value = y_pred_inf)