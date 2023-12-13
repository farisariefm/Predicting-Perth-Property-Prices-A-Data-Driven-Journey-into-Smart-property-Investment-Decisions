'''
===============================================================

Predicting Perth Property Prices: A Data-Driven Journey into Smart property Investment Decisions

Prepared by: Faris Arief Mawardi

Dataset: perth_houses.csv

Objective: Creating the homepage of the model deployment

'''

# Importing necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import eda
import models

# Initializing a selection box on the homepage sidebar
page = st.sidebar.selectbox(label='Select Page:', options=['Home Page', 'Exploratory Data Analysis', 'Prediction Model'])

# Initializing if loops to display content based on the selected page (home page, EDA, or modeling)
if page == 'Home Page':  # Initializing the contents of the homepage
    st.image('code_waves.jpg')
    st.markdown("<h1 style='text-align: center;'>Predicting Perth Property Prices: A Data-Driven Journey into Smart property Investment Decisions</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;font-style: italic;'>Prepared by : Faris Arief Mawardi</h1>", unsafe_allow_html=True)
    st.write("Objective: Develop a model to predict property prices in the Perth area aims to provide valuable insights into the factors influencing property values within the region.")
    st.write("Dataset: perth_houses.csv")
    st.markdown('[Link to Dataset](https://www.kaggle.com/datasets/syuzai/perth-house-prices)')
    st.write('')
    st.caption("Ready to uncover more layers of this project's intricacies? Navigate through the myriad of possibilities waiting for you in the sidebar menu!")
    st.write('')
    st.write('')
    with st.expander("**Background**"):  # Initializing a new section to explain the background
        st.write("<p style='text-align: center; font-weight: bold; font-style: italic;'>Digging Deeper Into the Property World!</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: justify;'>In the world of real estate in Perth, Western Australia, accurate property valuation is key to making smart decisions. This project aims to develop a property price prediction model with the hope of providing accurate estimation of property prices located in Perth. The project's goal is to provide stakeholders in the property industry with insights that can assist them in making informed decisions to optimize property investments in the Perth area based on accurate property price predictions.</p>", unsafe_allow_html=True)

    with st.expander("**Problems**"):  # Initializing a new section to explain the problem constraints addressed in this project
        st.text('1. What are the factors influencing property prices in Perth? \n2. Can the property construction year and the last selling year influence property prices in Perth? \n3. Can the location of a property affect its price? \n4. How will the property price prediction model in Perth be developed?')    
    with st.expander("**Problem Statement**"):  # Initializing a new section to discuss the problem statement
        st.write("<p style='text-align: justify;'>In the property industry, accurate property valuation is crucial for intelligent decision-making. Therefore, this project aims to develop a property price prediction model that can estimate property values in Perth, Western Australia. The primary goal of this research is to enable stakeholders in the property industry to make more informed decisions for maximizing property investment value in the Perth region through accurate property price predictions.</p>", unsafe_allow_html=True)
    with st.expander('**Workflow**'):  # Initializing a new section to discuss the workflow followed in this project
        st.image('Project_flowchart.jpg')
    with st.expander("**Conclusion**"):  # Initializing a new section to discuss the conclusions of this project
        st.markdown(
        '''
        1. Factors that can affect the price of a property in Perth can be observed through the correlation level of each factor with the property price. Several factors show significant correlation with the price:
            - **SUBURB**
            - **NEAREST_STN**
            - **NEAREST_SCH**
            - **BEDROOMS**
            - **LAND_AREA**
            - **CBD_DIST**
            - **NEAREST_SCH_RANK**

        2. Based on the investigation, the year of construction and the last sale year of a property do not exhibit a significant correlation that can influence property prices.

        3. The location of a property has a very strong correlation with the property price. Further research reveals a segmentation of suburbs into exclusive residential areas (e.g., Dalkieth, Floreat, Watermans Bay) and more affordable areas (e.g., Merriwa, Bertram, Butler). This segmentation indicates variations in property types and available facility qualities within these suburbs. Additionally, the distance from a location to the city center strongly correlates with differences in property prices based on the property's location.

        4. To conduct this project, supervised machine learning methods were employed to predict property prices based on influencing factors. The supervised machine learning models used in this project include KNN Regressor, SVM Regressor, Random Forest Regressor, Decision Tree Regressor, and Adaboosting Regressor. Based on the evaluation and tuning of models, the KNN Regressor method using independent variables **SUBURB**, **NEAREST_STN**, **NEAREST_SCH**, **BEDROOMS**, **LAND_AREA**, **CBD_DIST**, **NEAREST_SCH_RANK** demonstrates the most optimal model performance after evaluation using R-squared and MAE metrics.''')
        
elif page == 'Exploratory Data Analysis': # Menginisiasi tampilan untuk page Explaration Data Analysis 
    eda.run() # Running file EDA
     
else:
    models.run() # Running file models untuk page models