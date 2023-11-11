'''
===============================================

Faris Arief Mawardi

Dataset : perth_houses.csv

Dataset Source : https://www.kaggle.com/datasets/syuzai/perth-house-prices

Objective : Membuat Model deployment untuk project yang telah dibuat

'''


import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import json

def run():
# Model Loading
    with open('model.pkl', 'rb') as file_1:
        model = pickle.load(file_1)

        ADDRESS = st.text_input(label='Masukkan alamat properti yang ingin Anda prediksi')
        SUBURB = st.text_input(label='Masukkan Suburb/Daerah properti yang ingin Anda prediksi')
        BEDROOMS = st.number_input(label='Masukkan jumlah kamar tidur pada properti yang ingin Anda prediksi', min_value=1)
        BATHROOMS = st.number_input(label='Masukkan jumlah kamar mandi pada properti yang ingin Anda prediksi', min_value=1)
        GARAGE = st.number_input(label='Masukkan kapasitas maksimal jumlah kendaraan pada garasi properti yang ingin Anda prediksi', min_value=0)
        LAND_AREA = st.number_input(label='Masukkan luas tanah dari properti yang ingin Anda prediksi', min_value=1)
        FLOOR_AREA = st.number_input(label='Masukkan luas bangunan dari properti ingin Anda prediksi', min_value=1)
        BUILD_YEAR = st.number_input(label='Masukkan tahun dibangunnya properti yang ingin Anda prediksi', min_value=1)
        CBD_DIST = st.number_input(label='Masukkan jarak dari properti anda menuju ke pusat kota', min_value=0)
        NEAREST_STN = st.text_input(label='Masukkan Stasiun Kereta Terdekat dari properti anda')
        NEAREST_STN_DIST = st.number_input(label='Masukkan jarak dari properti anda menuju stasiun kereta terdekat', min_value=0)
        DATE_SOLD = st.number_input(label='Masukkan Tahun Penjualan terakhir dari properti anda', min_value=0)
        POSTCODE = st.number_input(label='Masukkan kode post alamat properti anda', min_value=0)
        LATITUDE = st.number_input(label='Masukkan latitude lokasi properti anda')
        LONGITUDE = st.number_input(label='Masukkan longitude lokasi properti anda')
        NEAREST_SCH = st.text_input(label='Masukkan nama sekolah terdekat dari properti anda')
        NEAREST_SCH_DIST = st.number_input(label='Masukkan jarak dari properti anda menuju ke sekolah terdekat', min_value=0)
        NEAREST_SCH_RANK = st.number_input(label='Masukkan peringkat sekolah terdekat dari properti anda', min_value=0)


    
    
    st.write('In the following is the result of the data you have input : ')
    
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
    
    if st.button(label='predict'):
    
        # Melakukan prediksi data dummy
        y_pred_inf = model.predict(data_inf)

        
        st.metric(label="Prediksi Harga Properti yang Anda Inginkan adalah ", value = y_pred_inf)