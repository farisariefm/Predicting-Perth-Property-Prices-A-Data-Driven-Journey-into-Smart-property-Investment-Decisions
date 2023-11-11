'''
===============================================

Faris Arief Mawardi

Dataset : perth_houses.csv

Objective : Membuat homepage deployment dari model prediksi yang telah dibuat

'''


import streamlit as st
import numpy as np
import pandas as pd
import eda
import models

# Menginisiasi sebuah selection box pada side bar homepage
page = st.sidebar.selectbox(label='Select Page:', options=['Home Page', 'Exploration Data Analysis', 'Model Prediksi'])

# Menginisiasi if loops untuk proses seleksi apa yang ditampilkan berdasarkan page yang dipilih (home page, EDA, atau modelling)
if page == 'Home Page': # Menginisiasi isi dari homepage
    st.title("Code Waves") 
    st.image('code_waves.jpg')
    st.title("Home Page") 
    st.header("Welcome to My 2nd Milestone !")
    st.text("Milestone - 2")
    st.text("By : Faris Arief Mawardi \nHCK - 009")
    st.write("Objective : Membuat model untuk memprediksi harga properti di wilayah Perth untuk menyediakan wawasan berharga tentang faktor-faktor yang memengaruhi harga properti di wilayah tersebut, serta memfasilitasi pengambilan keputusan yang lebih baik dalam pembelian dan penjualan properti di wilayah tersebut.")
    st.write("Dataset : perth_houses.csv")
    st.markdown('[Link Dataset](https://www.kaggle.com/datasets/syuzai/perth-house-prices)')
    st.write('')
    st.caption("Ready to uncover more layers of this project's intricacies? Navigate through the myriad of possibilities waiting for you in the sidebar menu!")
    st.write('')
    st.write('')
    with st.expander("**Latar Belakang**"): # Menginisiasi section baru untuk menjelaskan latar belakang
        st.write("<p style='text-align: center; font-weight: bold; font-style: italic;'>Menggali Lebih Dalam Menuju Dunia Properti!</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: justify;'>Dalam dunia properti di Perth, Australia Barat, ketepatan penilaian properti menjadi kunci untuk pengambilan keputusan yang cerdas. Proyek ini bertujuan untuk menciptakan model prediksi harga properti dengan harapan memberikan perkiraan yang akurat terkait harga properti yang terletak di Perth. Harapan dari proyek ini adalah memberikan pemangku kepentingan di industri properti sebuah wawasan yang dapat membantu mereka membuat keputusan yang bijak untuk mengoptimalkan investasi properti di wilayah Perth dengan pertimbangan yang cermat berdasarkan prediksi harga properti yang akurat.</p>", unsafe_allow_html=True)

    with st.expander("**Problems**"): # Menginisiasi section baru untuk menjelaskan batasan masalah yang akan dibahas pada project ini
        st.text('1. Apa saja faktor yang mempengaruhi harga suatu properti di Perth? \n2. Apakah tahun pembangunan properti dan tahun terakhir kali properti tersebut \ndijual dapat mempengaruhi harga suatu properti di Perth?\n3. Apakah letak suatu properti bisa mempengaruhi harga dari properti tersebut?\n4. Bagaimana model prediksi harga properti di Perth akan dibuat?')    
    with st.expander("**Problem Statement**"): # Menginisiasi section baru untuk membahas problem statement
        st.write("<p style='text-align: justify;'>Dalam industri properti wilayah Perth, Australia Barat, penilaian harga properti yang akurat menjadi kunci untuk pengambilan keputusan yang cerdas. Oleh karena itu, proyek ini bertujuan untuk mengembangkan model prediksi harga properti yang dapat memberikan estimasi harga suatu properti yang terletak di Perth. Tujuan utama dari penelitian ini adalah agar para pemangku kepentingan di industri properti dapat membuat keputusan yang lebih tepat dalam mengoptimalkan nilai investasi properti di wilayah Perth melalui pertimbangan prediksi harga properti yang akurat.</p>", unsafe_allow_html=True)
    with st.expander('**Flowchart Pengerjaan**'): # Menginisiasi section baru untuk membahas flowchart pengerjaan yang dilakukan pada project ini
        st.image('Project_flowchart.jpg')
    with st.expander("**Kesimpulan**"): # Menginisiasi section baru yang akan membahas kesimpulan dari project ini
        st.write("1. Faktor-faktor yang dapat mempengaruhi harga suatu properti di Perth dapat dilihat dari tingkat korelasi suatu fokter terhadap harga properti tersebut, beberapa faktor yang memiliki korelasi yang signifikan terhadap harga adalah :")
        st.write("<p style='font-weight: bold;'>- Daerah Properti </p> ", unsafe_allow_html=True) 
        st.write("<p style='font-weight: bold;'>- Bagaimana Fasilitas stasiun kereta terdekat dari properti tersebut </p>", unsafe_allow_html=True) 
        st.write("<p style='font-weight: bold;'>- Bagaimana fasilitas sekolah terdekat dari properti tersebut </p> ", unsafe_allow_html=True)
        st.write("<p style='font-weight: bold;'>- Berapa jumlah kamar tidur pada properti tersebut </p>", unsafe_allow_html=True)
        st.write("<p style='font-weight: bold;'>- Berapa luas tanah properti tersebut </p>", unsafe_allow_html=True) 
        st.write("<p style='font-weight: bold;'>- Jarak menuju pusat kota </p>", unsafe_allow_html=True) 
        st.write("<p style='font-weight: bold;'>- Peringkat sekolah yang terdekat dari properti tersebut </p>", unsafe_allow_html=True)
        st.write('')
        st.write("2. <p style='text-align: justify;'> Berdasarkan penelusuran yang dilakukan, tahun pembangunan serta tahun penjualan terakhir suatu properti tidak memiliki korelasi yang cukup signifikan untukn mempengaruhi harga suatu properti</p>", unsafe_allow_html=True)
        st.write('')
        st.write("<p style='text-align: justify;'> 3. Letak / daerah lokasi suatu properti memiliki korelasi yang sangat kuat terhadap harga properti. Setelah penelitian lebih lanjut, ditemukan adanya segmentasi suburb yang terbagi menjadi daerah hunian ekslusif (contoh : Dalkieth, Floreat, Watermans Bay), dan daerah dengan harga yang lebih terjangkau (contoh : Merriwa, Bertram, Butler). Segmentasi ini membuat adanya perbedaan jenis properti serta kualitas fasilitas yang tersedia di suburb tersebut. Selain itu, jarak suatu lokasi ke pusat kota juga terbukti memiliki korelasi yang kuat terhadap adanya perbedaan harga properti berdasarkan lokasi suatu properti tersebut.</p>", unsafe_allow_html=True)
        st.write('')
        st.write("<p style='text-align: justify;'> 4. Untuk menjalankan project ini, dilakukan metode supervised machine learning untuk memprediksi harga suatu properti berdasarkan faktor-faktor yang dapat mempengaruhinya. Metode/model supervised machine learning yang dilakukan pada project ini adalah KNN Regressor, SVM Regressor, Random Forest Regressor, Decision Tree Regressor, dan Adaboosting Regressor. Berdasarkan hasil evaluasi serta tuning model yang dilakukan, didapatkan hasil bahwa metode KNN Regressor dengan menggunakan variabel independent yang telah disebutkan di point kesimpulan nomor 1 dapat menghasilkan performa model yang paling optimal setelah dievaluasi dengan menggunakan R2 score.</p>", unsafe_allow_html=True)
    

elif page == 'Exploration Data Analysis': # Menginisiasi tampilan untuk page Explaration Data Analysis 
    eda.run() # Running file EDA
     
else:
    models.run() # Running file models untuk page models


