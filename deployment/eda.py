'''
===============================================================

Predicting Perth Property Prices: A Data-Driven Journey into Smart property Investment Decisions

Prepared by: Faris Arief Mawardi

Dataset: perth_houses.csv

Objective: Creating the homepage of the model deployment

'''
# Import the necessary library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Initiating the function to connect the homepage and the EDA
def run():
    st.title('Welcome to Exploratory Data Analysis')
# Loading the Dataset
    df= pd.read_csv('perth_houses.csv')
# Displaying the head and tail of the dataset
    st.header ('Dataset Review')    
# Print the first 10 entries of the dataset
    with st.expander('**First 10 entries of the dataset**'):
        st.table(df.head(10))
# Print the last 10 entries of the dataset
    with st.expander('**Last 10 entries of the dataset**'):
        st.table(df.tail(10))
# Initiating the header of the features explanation section
    st.header('Features of the dataset')
    with st.expander('**Features Explanations :**'):
        st.markdown('''
**Insights :**

The dataset used in this study comprises property data from the Perth area, Western Australia. This dataset includes various attributes associated with properties, such as the number of bedrooms, bathrooms, land size, floor area, build year, distance to the city center, proximity to the nearest train station and school, property price, and more. The dataset consists of 33,656 entries and 19 columns (attributes).

**Relevance to the Research Objectives:**

1. **Building a Model to Predict Property Prices in the Perth Area:** This dataset serves as the foundation for developing a property price prediction model. Relevant attributes like the number of bedrooms, land size, among others, will be used as features to train the model. Property price will be the target variable to predict.
  
2. **Analyzing Factors Influencing Property Prices:** The dataset encompasses various attributes that can serve as factors influencing property prices. For instance, analyzing the extent to which the number of bedrooms or build year impacts property prices can help identify the most significant attributes affecting property prices in the Perth area.

**Dataset Attribute Information:**

The dataset consists of 19 columns, and here is a brief explanation of each column:
1. **ADDRESS**: Property address, indicating the physical address of the property in the Perth area.
2. **SUBURB**: Suburb or area in the Perth region where the property is located.
3. **PRICE**: Property price, which is the target variable to be predicted by the model.
4. **BEDROOMS**: Number of bedrooms in the property.
5. **BATHROOMS**: Number of bathrooms in the property.
6. **GARAGE**: Number of garages/vehicle capacity that can be accommodated in the garage.
7. **LAND_AREA**: Land area of the property in square meters.
8. **FLOOR_AREA**: Floor area of the property in square meters.
9. **BUILD_YEAR**: Year of property construction.
10. **CBD_DIST**: Distance of the property to the Central Business District in meters (assumed).
11. **NEAREST_STN**: Name of the nearest train station from the property.
12. **NEAREST_STN_DIST**: Distance of the property to the nearest train station in meters (assumed).
13. **DATE_SOLD**: Date of property sale.
14. **POSTCODE**: Property area postcode.
15. **LATITUDE**: Latitude coordinates of the property.
16. **LONGITUDE**: Longitude coordinates of the property.
17. **NEAREST_SCH**: Name of the nearest school from the property.
18. **NEAREST_SCH_DIST**: Distance of the property to the nearest school in meters (assumed).
19. **NEAREST_SCH_RANK**: Rank of the nearest school.

**Data Modifications**

1. **BUILD_YEAR**: This column needs to be converted to numeric data type (integer) to facilitate analysis of the impact of the property's build year on its price.
2. **DATE_SOLD**: This column needs to be converted to a datetime data type.

Apart from these two columns, the remaining columns are already in the appropriate data format.''')
# Menampilkan hasil uji distribusi data
    st.header('Exploratory Data Analysis Results')
    with st.expander ('**Data Distributions Analysis**'):
        st.markdown('**Data Distributions Analysis**')
        st.image('analisis_distribusi.png')
        st.markdown('''
**Insights :**

From the descriptive statistical analysis, skewness, kurtosis, and visualization of data distribution, the following insights can be drawn:

1. The "PRICE" column shows a right-skewed (positively skewed) data distribution with a skewness around 1.78, indicating the presence of houses with significantly high prices.
2. The "BEDROOMS" column has a more even distribution with a mean around 3.66 and skewness approaching 0, indicating that the number of bedrooms in houses is more evenly distributed.
3. The "BATHROOMS" column also has a more even distribution with skewness close to 0, suggesting that the number of bathrooms in houses tends to be evenly distributed.
4. The "GARAGE" column has very high skewness, approximately 16.28, indicating the presence of many outliers with an extremely high number of garages.
5. The "LAND_AREA" column has a data distribution with a mean around 2740.64 and skewness around 33.76, suggesting the presence of some houses with very large land areas.
6. The "FLOOR_AREA" column has a more even data distribution with skewness approaching 0 (1.35), suggesting that the floor area of houses is relatively evenly distributed.
7. The "BUILD_YEAR" column has negative skewness, around -1.39, indicating the presence of some older construction years.
8. The "CBD_DIST" column has a skewness of about 0.88, indicating a more even data distribution for the distance towards the city center.
9. The "NEAREST_STN_DIST" column has a skewness around 2.45, indicating the presence of some very distant distances to stations.
10. The "POSTCODE" column has a skewness around 2.07, indicating the presence of some postal codes far from the average.
11. The "LATITUDE" column has skewness around -0.38 and kurtosis around -0.11, indicating a tendency toward symmetrical distribution.
12. The "LONGITUDE" column has skewness around 0.63 and kurtosis around 0.05, suggesting a tendency toward symmetrical distribution.
13. The "NEAREST_SCH_DIST" column has skewness around 3.75 and kurtosis around 20.03, indicating some very distant distances to schools.
14. The "NEAREST_SCH_RANK" column has skewness around 0.04 and kurtosis around -1.22, indicating a relatively symmetrical distribution.

From these insights, we can observe the differences and similarities in the characteristics of data distribution, skewness, kurtosis, and indications of outliers. The first set of data (descriptive statistics) shows several attributes with more even distributions, while the second set of data (skewness and kurtosis) indicates several attributes with skewed distributions. Additionally, the skewness and kurtosis data also suggest some attributes with higher skewness and kurtosis values, indicating longer tails in the data distribution and potential existence of outliers. Furthermore, after the indication of outliers in some attributes, an outlier analysis will be conducted to gain a more comprehensive understanding of the outliers present in the dataset.''')
    
    with st.expander ('**Outliers Identifications Analysis**'):
        st.markdown('**Outliers Identification Analysis in Numerical Features**')
        st.image('boxplots.png')
        st.markdown('''
**Insights :**

1. **PRICE:** Approximately 6.28% outliers are identified above the upper bound, indicating some properties with exceptionally high prices.
2. **BEDROOMS:** The percentage of outliers is relatively low, about 0.28% below the lower bound and 1.12% above the upper bound. This suggests a small number of properties with either fewer or more bedrooms compared to most properties.
3. **BATHROOMS:** Only about 0.80% of outliers are above the upper bound, suggesting a more evenly distributed data. However, there is a small proportion of properties with a higher number of bathrooms compared to the majority.
4. **GARAGE:** This attribute exhibits a remarkably high percentage of outliers, approximately 15.72% below the lower bound and 15.34% above the upper bound. This indicates a highly uneven data distribution, suggesting many properties having garages accommodating significantly more or fewer vehicles compared to the majority of properties that fall within the average range (2 cars).
5. **LAND_AREA:** There are about 14.58% outliers above the upper bound, indicating a significant number of properties with extremely large land areas compared to others.
6. **FLOOR_AREA:** Only about 2.23% of outliers are above the upper bound, suggesting a small proportion of properties with larger floor areas compared to others.
7. **BUILD_YEAR:** Outliers are identified with 2.87% below the lower bound, indicating some properties built in older years compared to the majority.
8. **CBD_DIST:** There are about 1.96% outliers above the upper bound, indicating some properties having a farther distance to the city center compared to most properties.
9. **NEAREST_STN_DIST:** Indicates a high percentage of outliers, approximately 9.01% above the upper bound, suggesting some properties have very distant distances to the train station.
10. **POSTCODE:** Only about 0.47% of outliers are above the upper bound.
11. **LATITUDE:** Low outlier percentages, with 1.19% below the lower bound and 0.25% above the upper bound.
12. **LONGITUDE:** Approximately 0.48% outliers are above the upper bound.
**Outliers in latitude and longitude might not be as significant due to values being confined within Perth's territorial boundaries.**
13. **NEAREST_SCH_DIST:** Outliers are around 6.82% above the upper bound, indicating some properties have very distant distances to schools.
14. **NEAREST_SCH_RANK:** No outliers are identified in this attribute.

With this information on the percentage of outliers, we can better understand the extent of extreme data points in each attribute and recognize the importance of further handling outliers.''')
        st.markdown('**Outliers Identification Analysis after Handling with Winsorizers Capping Method**')
        st.markdown('''
| Variabel             | Lower Bound Outliers | Upper Bound Outliers |
|----------------------|----------------------|----------------------|
| BEDROOMS             | 0.00%                | 0.00%                |
| GARAGE               | 18.40%               | 12.29%               |
| LAND_AREA            | 0.00%                | 0.00%                |
| CBD_DIST             | 0.00%                | 0.00%                |
| NEAREST_SCH_RANK     | 0.00%                | 0.00%                |
                    ''')
        st.markdown('''
After implementing the winsorizer as the method to address the issue of outliers, the results indicate that outliers in several features, such as BEDROOMS, BATHROOMS, and others, have been effectively handled. However, the 'GARAGE' feature still exhibits a notable percentage of outliers even after the applied treatment. This anomaly is presumed to persist, prompting further analysis involving an in-depth investigation into its unique values and necessitating additional treatment.''')
        st.text('')
        st.markdown('**Further Analysis on Outliers in Garage Features**')
        st.markdown('''
| Variabel   | Lower Bound Outliers | Upper Bound Outliers | Lower Boundary | Upper Boundary |
|------------|----------------------|----------------------|-----------------|-----------------|
| GARAGE     | 18.40%               | 12.29%               | 2.0             | 2.0             |
                    ''')
        st.image('boxplot garage.png')
        st.markdown('''
**Insight:**
Based on the provided data, it's evident that the values of the lower and upper boundaries for the 'garage' feature are identical. This observation is complemented by the box plot, indicating an extremely narrow quartile range (Q1-Q3), implying low skewness and high kurtosis within this garage data. Consequently, we'll consider the data in the 'garage' feature as invalid, potentially due to misinformation during the data collection process. For instance, individuals residing in apartments may not have personal garages since it's a public facility. Consequently, the 'garage' feature will be excluded from the dataset to prepare for model construction.''')
    
    with st.expander('**Features Correlations Analysis**'):
        st.markdown('**Numerical Features Correlations Analysis**')
        st.image('korelasi_numerik.png')
        st.markdown('''
**Insight:**

Based on the Kendall correlation matrix analysis, the correlation between the numeric attributes in the dataset against the target variable (price) has been assessed. The obtained correlation values are then categorized into several groups as follows:

- "High_Positive_Correlation" indicates a strong positive correlation (correlation > 0.5).
- "Moderate_Positive_Correlation" indicates a moderate positive correlation (0.5 < correlation > 0.1).
- "Low_Positive_Correlation" indicates a weak positive correlation (0.1 < correlation > 0).
- "High_Negative_Correlation" indicates a strong negative correlation (correlation < -0.5).
- "Moderate_Negative_Correlation" indicates a moderate negative correlation (-0.5 < correlation < -0.1).
- "Low_Negative_Correlation" indicates a weak negative correlation (correlation < -0.1).

Here are insights regarding the correlation of variables in the dataset with the "PRICE" variable:

1. The property price does not have a strong positive correlation with other features in the dataset.

2. Price has a moderate positive correlation with:
   - Number of bedrooms (**BEDROOMS**).
   - Number of bathrooms (**BATHROOMS**).
   - Garage capacity (**GARAGE**).
   - Land area of the property (**LAND_AREA**).
   - Floor area (**FLOOR_AREA**).

3. Price has a weak positive correlation with:
   - Latitude (**LATITUDE**).
   - Distance to the nearest school (**NEAREST_SCH_DIST**).
   - Property sale year (**DATE_SOLD**).

4. The property price does not have a strong negative correlation with other features in the dataset.

5. Price has a moderate negative correlation with:
   - Distance of the property to the city center (**CBD_DIST**).
   - Postal code (**POSTCODE**).
   - Longitude (**LONGITUDE**).
   - Nearest school's ranking from the property (**NEAREST_SCH_RANK**).

6. Price has a weak negative correlation with:
   - Year the property was built (**BUILD_YEAR**).
   - Distance to the nearest station (**NEAREST_STN_DIST**).

From the categorization of the numeric feature correlations with the price, we can conclude that several features might have a significant influence on the model to be built, as they exhibit moderate to high correlations. Hence, the features with low correlation that have redundant information or can be represented by other features will be removed from the dataset during the modeling preparation. Some features to be removed include:
   - Year the property was built (**BUILD_YEAR**) -> Insignificant correlation
   - Distance to the nearest station (**NEAREST_STN_DIST**) -> Insignificant correlation
   - Latitude (**LATITUDE**) -> Insignificant correlation
   - Longitude (**LONGITUDE**) -> Moderate correlation, but its function can be represented by suburb or address
   - Distance to the nearest school (**NEAREST_SCH_DIST**) -> Insignificant correlation
   - Property sale year (**DATE_SOLD**) -> Insignificant correlation
   - Postal code (**POST_CODE**) -> Moderate correlation, but its function can be represented by suburb or address
''')
        st.text('')
        st.markdown('**Categorical Features Correlations Analysis**')
        st.image('korelasi_kategorikal.png')
        st.markdown('''
**Insight:**

Based on the Phik correlation matrix analysis conducted, the correlation analysis between categorical attributes within the dataset and the target variable (price) was obtained. The correlation values were categorized into several groups as follows:

- "High_Positive_Correlation" represents a strong positive correlation (correlation > 0.5).
- "Moderate_Positive_Correlation" indicates a moderate positive correlation (0.5 < correlation > 0.1).
- "Low_Positive_Correlation" signifies a weak positive correlation (0.1 < correlation > 0).
- "High_Negative_Correlation" denotes a strong negative correlation (correlation < -0.5).
- "Moderate_Negative_Correlation" indicates a moderate negative correlation (-0.5 < correlation > -0.1).
- "Low_Negative_Correlation" signifies a weak negative correlation (correlation < -0.5).

Here are the insights regarding the correlation of categorical variables in the dataset with the "PRICE" variable:

1. Price has a strong positive correlation with:
   - Property location suburb (**SUBURB**)
   - Nearest station to the property (**NEAREST_STN**)
   - Nearest school to the property (**NEAREST_SCH**)

2. Price has a weak correlation with:
   - Property address (**ADDRESS**).

From the categorization of categorical feature correlations with the price, it can be concluded that features falling under moderate to high correlation categories tend to have a significant impact on the model. Therefore, features with low correlation will be removed from the dataset during the modeling preparation process. Based on the analysis of categorical feature correlations using Phik correlations, **ADDRESS** (Property address) is considered to have an insignificant correlation with property price. Hence, this feature will be eliminated.
''')
        st.text('')
        st.markdown('''
**Correlation Analysis Summary**

1. Price has a strong correlation with:
   - **SUBURB**: This indicates that the location or area where the property is situated is a primary factor in determining property prices. Certain suburbs might be more expensive or have higher-quality properties.
   - **NEAREST_STN** and **NEAREST_SCH**: Properties closer to train stations or nearby schools tend to have higher prices. Accessibility to transportation and education might be significant factors in determining property prices.

2. Price has a moderate positive correlation with:
   - **BEDROOMS**, **BATHROOMS**, **GARAGE**: Properties with more bedrooms, bathrooms, and garage capacity tend to have higher prices. This reflects higher demand for houses with more amenities.
   - **LAND_AREA** and **FLOOR_AREA**: The land area and floor area of the property also positively contribute to the price. Larger properties tend to have higher prices.
   - **DATE_SOLD**: The property's sale date also moderately correlates with the price. This might be related to market trends evolving over time.

3. Price has a weak positive correlation with:
   - **LATITUDE**: Weak positive correlation suggests a relationship between geographic location (latitude) and property prices.
   - **NEAREST_SCH_DIST**: Distance to the nearest school contributes weakly to the price, although not as strongly as other factors.

4. Price has a moderate negative correlation with:
   - **CBD_DIST**, **POSTCODE**, **LONGITUDE**, **NEAREST_SCH_RANK**: All these features contribute negatively to the price. Property distance to the city center, postcode, longitude, and the rank of the nearest school has a negative impact on property prices. Prices tend to be lower if the property is farther from the city center, has a lower postcode, or is located farther from high-quality schools.

5. Price has a weak negative correlation with:
   - **BUILD_YEAR**, **NEAREST_STN_DIST**, **ADDRESS**: Weak negative correlations suggest these features have a small influence on property prices.

Hence, this correlation analysis can aid potential property buyers or investors in understanding the factors influencing property prices in this area, enabling more informed decisions. Additionally, the correlation analysis results indicate some attributes that can be eliminated from the data preparation for further modeling, some of which include:

   - Year the property was built (**BUILD_YEAR**) -> Insignificant correlation
   - Distance to the nearest station (**NEAREST_STN_DIST**) -> Insignificant correlation
   - Latitude (**LATITUDE**) -> Insignificant correlation
   - Longitude (**LONGITUDE**) -> Moderate correlation, but its function can be represented by suburb or address
   - Distance to the nearest school (**NEAREST_SCH_DIST**) -> Insignificant correlation
   - Property sale year (**DATE_SOLD**) -> Insignificant correlation
   - Postcode (**POST_CODE**) -> Moderate correlation, but its function can be represented by suburb or address
   - Address (**ADDRESS**) -> Insignificant correlation                    ''')
        st.text('')
    with st.expander ('**Analysis of the Influence of Distance to the City Center on Property Prices**'):
        st.image('Analisis Pengaruh Jarak Menuju Pusat Kota terhadap Harga Properti.png')    
        st.markdown('''
**Insight:**

From the analysis correlating the distance of a property to the city center with its price, there is a tendency that the closer a property is to the city center, the higher its price tends to be. This indicates a moderate positive correlation between proximity to the city center and property prices. This finding suggests an increase in property prices for those closer to the city center. This information is valuable for prospective property buyers and sellers seeking to understand the influencing factors affecting property prices in that area.                    ''')
    with st.expander('**Analysis of the Top 10 Suburbs with the Highest Prices**'):
        st.image('map of perth.png')
        st.image('10 Suburbs Termahal.png')
        st.image('Jumlah Properti Top 10 Suburbs.png')
        st.markdown('''**Insight:**

In the context of comparing property availability, demand, and exclusivity (price) among exclusive suburbs (top 10 highest-priced) in Perth, we can classify suburbs into three categories based on their characteristics:

1. **Suburbs with High Demand**:
   - Suburbs such as "City Beach," "Floreat," "Watermans Bay," and "Mosman Park" exhibit a significant number of properties, indicating high demand in these areas. This high demand might be triggered by attractive geographical locations, access to public amenities, and the environmental allure.

2. **Exclusive Suburbs**:
   - Suburbs like "Applecross," "Hazelmere," and "Dalkeith," despite having fewer properties, boast high property prices, indicating the exclusivity of these areas where properties might be rarer and offer extremely exclusive amenities. This exclusivity can create high demand from specific market segments.

3. **Suburbs with Moderate Availability**:
   - Some other suburbs like "Peppermint Grove" also appear in the list of property numbers but are not as popular or exclusive as other suburbs. This might indicate that the demand in these areas is more moderate, and property availability is more balanced.

Therefore, it can be concluded that there are significant differences in property availability, demand, and the level of exclusivity among suburbs in Perth. Suburbs with high demand may reflect their popularity and appeal among potential property buyers. Exclusive suburbs with fewer properties may demonstrate high exclusivity, resulting in higher property prices. Suburbs with moderate availability create a balance between availability and demand.''')
    with st.expander('**Potential Suburbs for Further Development**'):
        st.image('Suburbs Potensial.png')
        st.image('Harga Suburbs Potensial.png')
        st.image('Perkembangan Pembangunan Properti Potensial.png')
        st.markdown('''**Insight:**

From the visualization of property development trends per year in the top 10 suburbs:

It is evident that suburbs such as "Butler," "Mindarie," and "Iluka" have shown a remarkably significant increase in property development over the last 5 years. This indicates a potential substantial business development in these areas due to the increased demand observed during this 5-year period.

However, suburbs like "Henley Brook," "Darch," and "Jane Brook" have experienced fluctuations tending towards a decrease in the number of properties in recent years, possibly due to certain factors influencing the property market in these areas.

Based on this analysis, we can conclude that some suburbs with high potential for development and increased demand are:

- Butler
- Mindarie
- Iluka''')
    
    with st.expander('**Multicollinearity Analisis**'):
        st.markdown('**Multicollinearity Analysis on Numerical Features**')
        st.markdown('''
| variabel           | VIF       |
|--------------------|-----------|
| BEDROOMS           | 26.733588 |
| BATHROOMS          | 17.790566 |
| GARAGE             | 3.182447  |
| LAND_AREA          | 1.049713  |
| FLOOR_AREA         | 12.419792 |
| CBD_DIST           | 4.396609  |
| NEAREST_SCH_RANK   | 5.585463  |

                    ''')
        st.markdown('''
**Insight:**

Considering the insights provided by the VIF data and the correlations between variables in the dataset, we can establish the following relationships:

**BEDROOMS and BATHROOMS:** Both variables exhibit high VIF values and a strong correlation. This signifies that "BEDROOMS" and "BATHROOMS" share significant multicollinearity and have a fairly strong positive correlation.

**FLOOR_AREA:** The variable "FLOOR_AREA" also demonstrates high VIF and strong correlation with some other variables such as "BEDROOMS" and "BATHROOMS."

From the above insights, to address multicollinearity within the dataset, dropping the "BATHROOMS" and "FLOOR_AREA" features is assumed to be a suitable step. It is believed that in practical scenarios, these two attributes might have less significant factors and can be represented by the "BEDROOMS" and "LAND_AREA" features, both in terms of information content and their correlation with property prices.
                    ''')
        st.markdown('**Multicollinearity After Dropping Floor Area dan Bathrooms Features**')
        st.markdown('''
| variabel           | VIF       |
|--------------------|-----------|
| BEDROOMS           | 8.007782  |
| GARAGE             | 3.120011  |
| LAND_AREA          | 1.045076  |
| CBD_DIST           | 4.395527  |
| NEAREST_SCH_RANK   | 5.467634  |
''')
        st.markdown('''
                    **Insight :**

Multicollinearity Issue have been successfully addressed.''')
        
    with st.expander('**Summary of EDA Process**'):
        st.markdown('''
**Conclusion of Exploratory Data Analysis**

Based on the Exploratory Data Analysis (EDA) conducted on the property dataset in Perth, we can draw several conclusions and insights as follows:

1. **Property Counts and Availability**: There is a variation in the number of properties across different suburbs in Perth. Suburbs such as "Bertram," "Iluka," "Bennett Springs," and "Mindarie" have a high number of properties, indicating dense populations and high demand for accommodation. However, there are also suburbs with fewer properties.

2. **Variation in Property Types**: There is diversity in the types of properties available in suburbs, which affects property prices. Some suburbs may offer more exclusive or premium property types, while others have more affordable properties.

3. **Property Investment**: This data can provide insights for property developers or potential investors about suburbs that might be attractive for further development or property investment according to their targeted investment segmentation (exclusive or more affordable properties). Insights based on population growth indicators (represented in this data by property development trends), property prices, and facilities in the surrounding area could guide potential investment decisions.

Additionally, other information observed about the dataset includes:
- Uneven data distribution in several attributes.
- Outliers: Significant indications of outliers in several attributes necessitate further outlier handling steps.
- Missing Values: Identified missing values in attributes like garage, build year, and nearest school rank, requiring further handling. The observations about these missing values also offer guidance on the appropriate steps for handling them based on the respective attributes:
    - Garage Attribute Missing Values: These missing values could be due to some properties not having a garage (Missing Not At Random) -> Missing values can be assumed as "0" or no garage.
    - Nearest School Rank: These missing values could be due to some missing informations, due to its significant impact towards property prices, the nearest school rank will be imputed by using the KNN imputer
    - Build Year: Missing values in this column are assumed to represent a lack of recorded or available information about the year of construction of the property. Given its extremely minimal correlation with property price, the approach to handle missing values in this attribute is by dropping the column from the data.
''')