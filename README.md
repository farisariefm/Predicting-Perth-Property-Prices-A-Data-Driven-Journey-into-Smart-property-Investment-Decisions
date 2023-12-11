# Predicting Perth Property Prices: A Data-Driven Journey into Smart property Investment Decisions
## Overview

This repository contains the code and analysis for predicting property prices in the Perth area, Western Australia. 
The primary goal is to develop a predictive model that provides estimates for property prices based on various influential factors. 
The insights from this project aim to facilitate informed decision-making in property investment and sales within the region.

Dataset Source : [Kaggle](https://www.kaggle.com/datasets/syuzai/perth-house-prices)

## Problem Statements
In the property industry, accurate property valuation is crucial for intelligent decision-making. Therefore, this project aims to develop a property price prediction model that can estimate property values in Perth, Western Australia. The primary goal of this research is to enable stakeholders in the property industry to make more informed decisions for maximizing property investment value in the Perth region through accurate property price predictions.

### Focused Approach
- Identify factors influencing property prices in Perth.
- Assess the impact of property development years and sale years on prices.
- Determine the significance of property location on pricing.

## Data Preparation and Preprocessing

### Data Identification
- Identifying relevant attributes like bedrooms, bathrooms, land area, built year, distance to city center, nearest station, and school proximity.

### Exploratory Data Analysis
- Initial Data Understandings
- Identifying missing values and Outliers
- Conducting descriptive statistics analytic for relevant property attributes.

## Preprocessing and Machine Learning Analysis

### Data Preprocessing
- Handling Missing Values and Outliers
- Features Scaling and Encoding
- Correlation tests to determine relationships between property attributes and prices.
- Calculating correlations to measure attribute-property relationships.

### Machine Learning Application
- Application of supervised machine learning regression methods for predicting property prices.
- Model evaluation and inference testing for accuracy assessment.

## Conclusion and Recommendations

### Influential Factors
- Suburb, nearest station, nearest school, bedrooms, and land area exhibit significant correlations with property prices.
- Insights on property development and recent sale years showed negligible impacts on property pricing.

### Location Significance
- Property location, segmented into exclusive and affordable areas, significantly affects pricing.
- Distance from the city center plays a pivotal role in property pricing variations.

## Results Summary

### Best-Performing Model
- Utilizing KNN Regressor with specific independent variables yielded the most optimal model performance based on R^2 score.
