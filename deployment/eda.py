'''
===============================================

Faris Arief Mawardi

Dataset : perth_houses.csv

Objective : Membuat homepage deployment dari model prediksi yang telah dibuat

'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Membuat function untuk menghubungkan eda.py dengan app.py
def run():
    st.title('Welcome to Exploratory Data Analysis')
# Memanggil data csv 
    df= pd.read_csv('perth_houses.csv')
# Menampilkan Review Dataset (10 teratas dan terbawah)
    st.header ('Dataset Review')    
# Menampilkan 10 data teratas
    with st.expander('**Menampilkan 10 Data Teratas pada Dataset**'):
        st.table(df.head(10))
# Menampilkan 10 data terbawah
    with st.expander('**Menampilkan 10 Data Terbawah pada Dataset**'):
        st.table(df.tail(10))
# Menampilkan header untuk penjelasan fitur (kolom-kolom) pada dataset
    st.header('Penjelasan Fitur Dataset')
    with st.expander('**Penjelasan Fitur Dataset**'):
        st.markdown('Dataset ini memiliki 19 kolom, dan berikut adalah penjelasan singkat tentang masing-masing kolom:\n1. **ADDRESS**: Alamat properti, menunjukkan alamat fisik properti yang ada di wilayah Perth.\n2. **SUBURB**: Suburb atau kawasan di wilayah Perth tempat properti berada.\n3. **PRICE**: Harga properti, merupakan variabel target yang akan diprediksi oleh model.\n4. **BEDROOMS**: Jumlah kamar tidur di properti.\n5. **BATHROOMS**: Jumlah kamar mandi di properti.\n6. **GARAGE**: Jumlah Garasi / kapasitas kendaraan yang dapat dimasukkan ke dalam garasi \n7. **LAND_AREA**: Luas tanah properti dalam satuan meter persegi.\n8. **FLOOR_AREA**: Luas bangunan properti dalam satuan meter persegi.\n9. **BUILD_YEAR**: Tahun pembangunan properti.\n10. **CBD_DIST**: Jarak properti ke pusat kota (Central Business District) dalam meter (diasumsikan).\n11. **NEAREST_STN**: Nama stasiun kereta terdekat dari properti.\n12. **NEAREST_STN_DIST**: Jarak properti ke stasiun kereta terdekat dalam meter (diasumsikan).\n13. **DATE_SOLD**: Tanggal penjualan properti.\n14. **POSTCODE**: Kode pos wilayah properti.\n15. **LATITUDE**: Koordinat lintang properti.\n16. **LONGITUDE**: Koordinat bujur properti.\n17. **NEAREST_SCH**: Nama sekolah terdekat dari properti.\n18. **NEAREST_SCH_DIST**: Jarak properti ke sekolah terdekat dalam meter (diasumsikan).\n19. **NEAREST_SCH_RANK**: Peringkat sekolah terdekat.')
# Menampilkan hasil uji distribusi data
    st.header('Exploratory Data Analysis Results')
    with st.expander ('**Analisis Distribusi Data**'):
        st.markdown('**Analisis Distribusi Data**')
        st.image('analisis_distribusi.png')
        st.markdown ('**Insight :**')
        st.markdown("Dari analisis statistik deskriptif, skewness, kurtosis, serta visualisasi distribusi data, dapat disimpulkan bahwa :\n1. Kolom 'PRICE' menunjukkan distribusi data yang condong ke kanan (positively skewed) dengan skewness sekitar 1,78, mengindikasikan adanya rumah-rumah dengan harga yang sangat tinggi.\n2. Kolom 'BEDROOMS' memiliki distribusi data yang lebih merata dengan mean sekitar 3,66 dan skewness mendekati 0, menunjukkan bahwa jumlah kamar tidur di rumah memiliki distribusi yang lebih merata.\n3. Kolom 'BATHROOMS' juga memiliki distribusi data yang lebih merata dengan skewness mendekati 0, menunjukkan bahwa jumlah kamar mandi di rumah cenderung terdistribusi secara merata.\n4. Kolom 'GARAGE' memiliki skewness yang sangat tinggi, sekitar 16,28. Hal ini mengindikasikan adanya banyak outliers dengan jumlah garasi yang sangat tinggi.\n5. Kolom 'LAND_AREA' memiliki distribusi data dengan mean sekitar 2740.64 dan skewness sekitar 33,76. Hal ini mengindikasikan adanya beberapa rumah dengan lahan yang sangat besar.\n6. Kolom 'FLOOR_AREA' memiliki distribusi data yang lebih merata dengan skewness mendekati 0 (1.35), menunjukkan bahwa luas lantai rumah terdistribusi dengan relatif lebih merata.\n7. Kolom 'BUILD_YEAR' memiliki skewness negatif, sekitar -1.39, mengindikasikan adanya beberapa tahun pembangunan yang lebih tua.\n8. Kolom 'CBD_DIST' memiliki skewness sekitar 0.88, menunjukkan distribusi data yang lebih merata untuk jarak ke pusat kota.\n9. Kolom 'NEAREST_STN_DIST' memiliki skewness sekitar 2.45, menunjukkan adanya beberapa jarak ke stasiun yang sangat jauh.\n10. Kolom 'POSTCODE' memiliki skewness sekitar 2.07, menunjukkan adanya beberapa kode pos yang jauh dari rata-rata.\n11. Kolom 'LATITUDE' memiliki skewness sekitar -0.38 dan kurtosis sekitar -0.11, menunjukkan distribusi data yang cenderung simetris.\n12. Kolom 'LONGITUDE' memiliki skewness sekitar 0.63 dan kurtosis sekitar 0.05, menunjukkan distribusi data yang cenderung simetris.\n13. Kolom 'NEAREST_SCH_DIST' memiliki skewness sekitar 3.75 dan kurtosis sekitar 20.03, menunjukkan adanya beberapa jarak ke sekolah yang sangat jauh.\n14. Kolom 'NEAREST_SCH_RANK' memiliki skewness sekitar 0.04 dan kurtosis sekitar -1.22, menunjukkan distribusi yang cenderung simetris.\nDari insight tersebut, kita dapat melihat perbedaan dan persamaan dalam karakteristik distribusi data, skewness, kurtosis, dan indikasi outliers. Data pertama (data statistik deskriptif) memiliki beberapa atribut dengan distribusi yang lebih merata, sedangkan pada data kedua (skewness dan kurtosis) menunjukkan beberapa atribut dengan distribusi yang memiliki kecondongan. Selain itu, pada data skewness dan kurtosis juga mengindikasikan adanya beberapa atribut dengan nilai skewness dan kurtosis yang lebih tinggi, menunjukkan adanya ekor yang lebih panjang pada distribusi data. Selanjutnya, setelah adanya indikasi outliers pada beberapa atribut, analisis outliers akan dilakukan untuk mendapatkan gambaran yang lebih lengkap terkait outliers yang terdapat pada dataset.")
    
    with st.expander ('**Analisis Outliers**'):
        st.markdown('**Analisis Outliers Fitur Numerik Dataset**')
        st.image('boxplots.png')
        st.markdown('**Insight :**')
        st.markdown("1. **PRICE**: Terdapat sekitar 6.28% outliers upper bound, menunjukkan adanya beberapa harga properti yang sangat tinggi.\n2. **BEDROOMS**: Persentase outliers terbilang rendah, sekitar 0.28% di bawah batas bawah (lower bound) dan 1.12% di atas batas atas (upper bound). Mengindikasikan adanya sejumlah kecil beberapa properti yang memiliki jumlah kamar lebih rendah ataupun lebih banyak dari sebagian besar properti lainnya.\n3. **BATHROOMS**: Hanya sekitar 0.80% outliers di atas batas atas (upper bound). Mengindikasikan distribusi data yang lebih merata, namun masih ada sebagian kecil properti yang memiliki jumlah bathrooms yang lebih tinggi dari properti lainnya\n4. **GARAGE**: Atribut ini memiliki persentase outliers yang sangat tinggi, sekitar 15.72% di bawah batas bawah (lower bound) dan 15.34% di atas batas atas (upper bound). Hal ini mengindikasikan bahwa distribusi data sangat tidak merata, sehingga ada cukup banyak properti memiliki garasi yang dapat menampung kendaraan lebih banyak ataupun lebih sedikit dibandingkan sebagian besar properti yang masih masuk ke dalam range rata-rata (2 mobil).\n5. **LAND_AREA**: Terdapat sekitar 14.58% outliers di atas batas atas (upper bound), menunjukkan adanya sebagian besar properti dengan lahan yang sangat besar dibandingkan beberapa rumah lainnya.\n6. **FLOOR_AREA**: Hanya sekitar 2.23% outliers di atas batas atas (upper bound). Menunjukkan adanya sebagian kecil properti yang memiliki luas bangunan yang lebih besar dibandingkan dengan properti lainnya.\n7. **BUILD_YEAR**: Persentase outliers ditemukan dengan 2.87% di bawah batas bawah, mengindikasikan beberapa properti yang dibangun pada range tahun yang lebih lama dibandingkan sebagian besar properti lainnya.\n8. **CBD_DIST**: Terdapat sekitar 1.96% outliers di atas batas atas (upper bound). Menunjukkan adanya sebagian properti yang memiliki jarak lebih jauh menuju pusat kota dibandingkan sebagian besar properti lainnya.\n9. **NEAREST_STN_DIST**: Mengindikasikan persentase outliers tinggi, sekitar 9.01% di atas batas atas (upper bound), menunjukkan terdapat beberapa jarak yang sangat jauh ke stasiun.\n10. **POSTCODE**: Hanya sekitar 0.47% outliers di atas batas atas (upper bound).\n11. **LATITUDE**: Persentase outliers rendah, dengan 1.19% di atas batas bawah dan 0.25% di atas batas atas.\n12. **LONGITUDE**: Terdapat sekitar 0.48% outliers di atas batas atas (upper bound).\n**Outliers latitude dan longitude mungkin tidak terlalu signifikan dikarenakan values nya hanya dibatasi pada teritorial perth**\n13. **NEAREST_SCH_DIST**: Persentase outliers sekitar 6.82% di atas batas atas (upper bound), menunjukkan adanya beberapa properti yang memiliki jarak yang sangat jauh ke sekolah.\n14. **NEAREST_SCH_RANK**: Tidak ada outliers yang diidentifikasi dalam atribut ini.\nDengan informasi persentase outliers ini, kita dapat lebih memahami sejauh mana sebaran data yang ekstrem dalam masing-masing atribut dan juga memberikan informasi perlunya dilakukan penanganan outliers lebih lanjut.")
        st.markdown('**Persentase Outliers Setelah Capping dengan Winsorizer**')
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
Dari perbandingan data outliers di atas, kita dapat melihat bahwa penanganan outliers telah mengurangi persentase outliers pada beberapa atribut, seperti BEDROOMS, BATHROOMS, dan beberapa atribut lainnya. Namun, atribut GARAGE tetap memiliki persentase outliers yang signifikan setelah penanganan. Hal ini diasumsikan sebagai anomali, namun akan dianalisis lebih lanjut dengan melakukan investigasi mendalam terkait unique values nya dan penanganannya lebih lanjut                    ''')
        st.text('')
        st.markdown('**Pengamatan Outliers pada Fitur Garage**')
        st.markdown('''
| Variabel   | Lower Bound Outliers | Upper Bound Outliers | Lower Boundary | Upper Boundary |
|------------|----------------------|----------------------|-----------------|-----------------|
| GARAGE     | 18.40%               | 12.29%               | 2.0             | 2.0             |
                    ''')
        st.image('boxplot garage.png')
        st.markdown('''
**Insight :**
Dari data di atas, dapat dilihat bahwa nilai lower boundary dan upper boundary dari fitur garage memiliki nilai yang sama, hal tersebut juga didukung oleh boxplot yang juga menggambarkan range quartile (Q1-Q3) yang sangat sempit, sesuai dengan skewness yang kecil dan kurtosis yang besar pada data garage ini. Oleh karena itu, Kita akan menganggap bahwa data pada fitur garage sebagai data yang tidak valid, dan akan dieliminasi dari data untuk persiapan pembuatan model.                    ''')
    
    with st.expander('**Analisi Korelasi Data**'):
        st.markdown('**Analisis Korelasi Fitur Numerik terhadap Prediksi Harga Properti**')
        st.image('korelasi_numerik.png')
        st.markdown('**Insight :**')
        st.markdown('''

Berdasarkan hasil Matriks korelasi kendall yang dilakukan, didapatkan hasil analisis korelasi antara atribut-atribut numerik dalam dataset terhadap variabel target (harga). Nilai korelasi yang didapat kemudian akan dikategorikan menjadi beberapa kelompok sebagai berikut :

- "High_Positive_Correlation" menunjukkan korelasi positif yang kuat (korelasi > 0.5).
- "Moderate_Positive_Correlation" menunjukkan korelasi positif yang sedang (0.5 < korelasi > 0.1).
- "Low_Positive_Correlation" menunjukkan korelasi positif yang lemah (0.1 < korelasi > 0).
- "High_Negative_Correlation" menunjukkan korelasi negatif yang kuat (korelasi > 0.5).
- "Moderate_Negative_Correlation" menunjukkan korelasi negatif yang sedang (0.5 < korelasi > 0.1).
- "Low_Negative_Correlation" menunjukkan korelasi negatif yang lemah (korelasi > 0.5).

Berikut adalah insight terkait korelasi variabel-variabel pada dataset terhadap variabel "PRICE":

1. Harga properti tidak memiliki korelasi positif yang kuat dengan fitur lainnya yang ada pada dataset

2. Harga Memiliki korelasi positif yang sedang dengan:
   - Jumlah kamar tidur (**BEDROOMS**).
   - Jumlah kamar mandi (**BATHROOMS**).
   - Jumlah/kapasitas kendaraan dalam garasi (**GARAGE**)
   - Luas Tanah Properti (**LAND_AREA**)
   - Luas Bangunan (**FLOOR_AREA**)

3. Harga memiliki korelasi positif yang lemah dengan:
   - Latitude (**LATITUDE**).
   - Jarak ke sekolah terdekat (**NEAREST_SCH_DIST**).
   - Tahun Penjualan Properti (**DATE_SOLD**)

4. Harga properti tidak memiliki korelasi negatif yang kuat dengan fitur lainnya yang ada pada dataset.

5. Harga Memiliki korelasi negatif yang sedang dengan:
   - Jarak Properti ke Pusat Kota (**CBD_DIST**)
   - Kode Pos (**POSTCODE**)
   - Longitude (**LONGITUDE**)
   - Peringkat sekolah terdekat dari properti (**NEAREST_SCH_RANK**)

6. Harga memiliki korelasi negatif yang lemah dengan:
   - Tahun properti tersebut dibangun (**BUILD_YEAR**).
   - Jarak ke stasiun terdekat (**NEAREST_STN_DIST**).

Dari hasil kategorisasi korelasi fitur numerik terhadap harga, kita dapat menyimpulkan bahwa beberapa fitur yang bisa memiliki pengaruh cukup signifikan pada model yang akan dibuat adalah fitur yang berkategori moderate to high correlations, sehingga fitur yang memiliki low correlation, dan memiliki informasi yang dapat direpresentasikan oleh fitur lainnya akan dihapus dari dataset pada proses persiapan modelling. Beberapa fitur yang akan dihapus adalah :
   - Tahun properti tersebut dibangun (**BUILD_YEAR**) -> Korelasi yang kurang signifikan
   - Jarak ke stasiun terdekat (**NEAREST_STN_DIST**) -> Korelasi yang kurang signifikan
   - Latitude (**LATITUDE**) -> Korelasi yang kurang signifikan
   - Longitude (**LONGITUDE**) -> Korelasi moderat, namun fungsinya dapat direpresentasikan oleh suburb ataupun address
   - Jarak ke sekolah terdekat (**NEAREST_SCH_DIST**) -> Korelasi yang kurang signifikan 
   - Tahun Penjualan Properti (**DATE_SOLD**) -> Korelasi yang kurang signifikan
   - Kode Pos (**POST_CODE**) -> Korelasi moderat, namun fungsinya dapat direpresentasikan oleh suburb ataupun address
                    ''')
        st.text('')
        st.markdown('**Analisis Korelasi Fitur Kategorikal terhadap Harga Properti**')
        st.image('korelasi_kategorikal.png')
        st.markdown('**Insight :**')
        st.markdown('''

Berdasarkan hasil Matriks korelasi phik yang dilakukan, didapatkan hasil analisis korelasi antara atribut-atribut kategorikal dalam dataset terhadap variabel target (harga). Nilai korelasi yang didapat kemudian akan dikategorikan menjadi beberapa kelompok sebagai berikut :

- "High_Positive_Correlation" menunjukkan korelasi positif yang kuat (korelasi > 0.5).
- "Moderate_Positive_Correlation" menunjukkan korelasi positif yang sedang (0.5 < korelasi > 0.1).
- "Low_Positive_Correlation" menunjukkan korelasi positif yang lemah (0.1 < korelasi > 0).
- "High_Negative_Correlation" menunjukkan korelasi negatif yang kuat (korelasi > 0.5).
- "Moderate_Negative_Correlation" menunjukkan korelasi negatif yang sedang (0.5 < korelasi > 0.1).
- "Low_Negative_Correlation" menunjukkan korelasi negatif yang lemah (korelasi > 0.5).

Berikut adalah insight terkait korelasi variabel-variabel kategorikal pada dataset terhadap variabel "PRICE":

1. Harga memiliki korelasi positif yang kuat dengan :
   - Kawasan tempat properti berada (**SUBURB**)
   - Stasiun terdekat dari properti (**NEAREST_STN**)
   - Sekolah terdekat dari properti (**NEAREST_SCH**)

2. Harga Memiliki korelasi positif yang sedang dengan:
   - Tanggal Properti Terjual terakhir (**DATE_SOLD**).
 
3. Harga memiliki korelasi negatif yang lemah dengan:
   - Alamat properti berada (**ADDRESS**).

Dari hasil kategorisasi korelasi fitur kategorikal terhadap harga, kita dapat menyimpulkan bahwa beberapa fitur yang bisa memiliki pengaruh cukup signifikan pada model yang akan dibuat adalah fitur yang berkategori moderate to high correlations, sehingga fitur yang memiliki low correlation akan dihapus dari dataset pada proses persiapan modelling. Berdasakan analisis korelasi fitur kategorikal yang dilakukan dengan menggunakan phik correlations, **ADDRESS** (Alamat properti) dianggap tidak memiliki korelasi yang signifikan terhadap harga properti, sehingga fitur tersebut akan dieliminasi.
                    ''')
        st.text('')
        st.markdown('**Kesimpulan Analisis Korelasi**')
        st.markdown('''1. Harga memiliki korelasi kuat terhadap:
   - **SUBURB**: Ini menunjukkan bahwa lokasi atau kawasan tempat properti berada adalah salah satu faktor utama dalam menentukan harga properti. Beberapa suburb mungkin lebih mahal atau memiliki kualitas properti yang lebih tinggi.
   - **NEAREST_STN** dan **NEAREST_SCH**: Properti yang berdekatan dengan stasiun kereta atau sekolah terdekat cenderung memiliki harga yang lebih tinggi. Aksesibilitas ke transportasi dan pendidikan mungkin menjadi faktor penentu yang signifikan dalam menentukan harga properti.

2. Harga memiliki korelasi positif yang sedang dengan:
   - **BEDROOMS**, **BATHROOMS**, **GARAGE**: Properti dengan lebih banyak kamar tidur, kamar mandi, dan kapasitas garasi cenderung memiliki harga yang lebih tinggi. Ini mencerminkan permintaan yang lebih tinggi untuk rumah dengan lebih banyak fasilitas.
   - **LAND_AREA** dan **FLOOR_AREA**: Luas tanah dan luas bangunan properti juga berkontribusi positif terhadap harga. Properti yang lebih besar cenderung memiliki harga yang lebih tinggi.
   - **DATE_SOLD**: Tanggal terjualnya properti juga memiliki korelasi positif yang sedang dengan harga. Hal ini mungkin terkait dengan tren pasar yang berubah seiring waktu.

3. Harga memiliki korelasi positif yang lemah dengan:
   - **LATITUDE**: Korelasi positif yang lemah menunjukkan bahwa terdapat hubungan antara letak geografis (garis lintang) dan harga properti.
   - **NEAREST_SCH_DIST**: Jarak ke sekolah terdekat berkontribusi positif yang lemah terhadap harga, meskipun tidak sekuat faktor lain.

4. Harga memiliki korelasi negatif yang sedang dengan:
   - **CBD_DIST**, **POSTCODE**, **LONGITUDE**, **NEAREST_SCH_RANK**: Semua fitur ini berkontribusi negatif terhadap harga. Jarak properti ke pusat kota, kode pos, letak geografis (garis bujur), dan peringkat sekolah terdekat memiliki pengaruh negatif pada harga properti. Harga cenderung lebih rendah jika properti berjarak jauh dari pusat kota, memiliki kode pos yang lebih rendah, atau terletak lebih jauh dari sekolah berkualitas.

5. Harga memiliki korelasi negatif yang lemah dengan:
   - **BUILD_YEAR**, **NEAREST_STN_DIST**, **ADDRESS**: Korelasi negatif yang lemah menunjukkan bahwa fitur-fitur ini memiliki pengaruh kecil terhadap harga properti.

Dengan demikian, analisis korelasi ini dapat membantu calon pembeli atau investor properti untuk memahami faktor-faktor yang memengaruhi harga properti di daerah ini dan membuat keputusan yang lebih terinformasi. Selain itu, hasil analisis korelasi ini juga mengindikasikan adanya beberapa atribut yang bisa dieliminasi dari persiapan data untuk modelling lebih lanjut, beberapa di antaranya adalah : 

   - Tahun properti tersebut dibangun (**BUILD_YEAR**) -> Korelasi yang kurang signifikan
   - Jarak ke stasiun terdekat (**NEAREST_STN_DIST**) -> Korelasi yang kurang signifikan
   - Latitude (**LATITUDE**) -> Korelasi yang kurang signifikan
   - Longitude (**LONGITUDE**) -> Korelasi moderat, namun fungsinya dapat direpresentasikan oleh suburb ataupun address
   - Jarak ke sekolah terdekat (**NEAREST_SCH_DIST**) -> Korelasi yang kurang signifikan 
   - Tahun Penjualan Properti (**DATE_SOLD**) -> Korelasi yang kurang signifikan
   - Kode Pos (**POST_CODE**) -> Korelasi moderat, namun fungsinya dapat direpresentasikan oleh suburb ataupun address
   - Address (**ADDRESS**) -> Korelasi yang kurang signifikan
                    ''')
        st.text('')
    with st.expander ('**Analisis Pengaruh Jarak Menuju Pusat Kota terhadap Harga Properti**'):
        st.image('Analisis Pengaruh Jarak Menuju Pusat Kota terhadap Harga Properti.png')    
        st.markdown('''**Insight:**

Dari analisis yang menghubungkan jarak suatu properti ke pusat kota dengan harga propertinya, terlihat adanya kecenderungan bahwa semakin dekat suatu properti ke pusat kota, harganya cenderung lebih tinggi. Hal ini mengindikasikan adanya korelasi positif moderat antara kedekatan dengan pusat kota dan harga properti. Penemuan ini dapat dijelaskan dengan adanya peningkatan harga properti yang lebih tinggi untuk properti yang lebih dekat dengan pusat kota. Ini adalah informasi berharga bagi calon pembeli dan penjual properti yang ingin memahami faktor-faktor yang memengaruhi harga properti di wilayah tersebut.
                    ''')
    with st.expander('**Analisis 10 Suburbs dengan Harga Tertinggi**'):
        st.image('map of perth.png')
        st.image('10 Suburbs Termahal.png')
        st.image('Jumlah Properti Top 10 Suburbs.png')
        st.markdown('''**Insight :**

Dalam konteks perbandingan ketersediaan jumlah properti, demand, dan eksklusivitas (harga) di antara suburbs eksklusif (10 harga tertinggi) di Perth, kita dapat mengklasifikasikan suburbs menjadi tiga kategori berdasarkan karakteristiknya:

1. **Suburbs dengan Permintaan Tinggi (High Demand)**:
   - Suburbs seperti "City Beach," "Floreat," "Watermans Bay," dan "Mosman Park" memiliki jumlah properti yang signifikan, menunjukkan permintaan yang tinggi di daerah ini. Permintaan tinggi ini mungkin dipicu oleh lokasi geografis yang menarik, akses ke fasilitas umum, dan daya tarik lingkungannya.

2. **Suburbs Eksklusif (Exclusive Suburbs)**:
   - Suburbs seperti "Applecross," "Hazelmere," dan "Dalkeith" meskipun memiliki jumlah properti yang lebih sedikit, harga properti yang tinggi mengindikasikan eksklusivitas daerah ini, di mana properti mungkin lebih langka dan menawarkan fasilitas yang sangat eksklusif. Eksklusivitas dapat menciptakan permintaan yang tinggi dari segmen pasar tertentu.

3. **Suburbs dengan Ketersediaan Menengah (Moderate Availability)**:
   - Beberapa suburbs lainnya seperti "Peppermint Grove" juga muncul dalam daftar jumlah properti, namun tidak sepopuler atau se-eksklusif suburbs lainnya. Hal ini dapat mengindikasikan bahwa permintaan di daerah ini mungkin lebih moderat, dan ketersediaan properti lebih seimbang.

Dengan demikian, dapat disimpulkan bahwa ada perbedaan yang signifikan dalam ketersediaan jumlah properti, permintaan, dan tingkat eksklusivitas antara suburbs di Perth. Suburbs yang memiliki permintaan tinggi dapat mencerminkan popularitas dan daya tariknya di antara calon pembeli properti. Suburbs eksklusif dengan jumlah properti yang lebih sedikit dapat menunjukkan eksklusivitas tinggi, yang mungkin menghasilkan harga properti yang lebih tinggi. Suburbs dengan ketersediaan menengah menciptakan keseimbangan antara ketersediaan dan permintaan.
                    ''')
    with st.expander('**Suburbs Dengan Potensi Pengembangan yang Besar**'):
        st.image('Suburbs Potensial.png')
        st.image('Harga Suburbs Potensial.png')
        st.image('Perkembangan Pembangunan Properti Potensial.png')
        st.markdown('''**Insight :**

Dari visualisasi perkembangan pembangunan properti per tahun pada 10 suburb teratas, 

Terlihat bahwa suburbs "Butler", "mindarie", dan "iluka" terdapat peningkatan pembangunan properti yang sangat signifikan pada 5 tahun terakhir, dan hal ini menunjukan indikasi potensi pengembangan bisnis yang cukup besar di wilayah tersebut dikarenakan adanya demand yang meningkat pada rentang 5 tahun terakhir.

Namun, suburb "Henley Brook," "Darch," dan "Jane Brook" mengalami fluktuasi yang cenderung mengarah ke penurunan jumlah properti dalam beberapa tahun terakhir, mungkin karena faktor-faktor tertentu yang memengaruhi pasar properti di wilayah-wilayah tersebut.

Berdasarkan analisis ini, kita dapat menyimpulkan bahwa beberapa suburb yang berpotensi untuk dikembangkan dan memiliki demand yang tinggi adalah :

- Butler
- Mindarie
- Iluka
                    ''')
    
    with st.expander('**Analisis Multicollinearity**'):
        st.markdown('**Multicollinearity Numerical Features pada Dataset**')
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
        st.markdown('''**Insight :**

Dengan mempertimbangkan insight yang diberikan oleh data VIF dan korelasi antara variabel-variabel dalam data, kita dapat membuat beberapa hubungan berikut:

**BEDROOMS dan BATHROOMS:** Kedua variabel ini memiliki VIF yang tinggi dan korelasi yang kuat. Ini berarti bahwa "BEDROOMS" dan "BATHROOMS" memiliki hubungan multikolinearitas yang signifikan dan memiliki korelasi positif yang cukup kuat. 

**FLOOR_AREA:** Variabel "FLOOR_AREA" memiliki VIF yang tinggi dan korelasi yang kuat dengan beberapa variabel lain seperti "BEDROOMS" dan "BATHROOMS."

Dari insight di atas, untuk menangani adanya multikolinearitas dalam data, amelakukan drop terhadap fitur bathrooms dan floor area diasumsikan menjadi langkah yang tepat karena dirasa pada kasus actual, kedua atribut di atas memiliki faktor yang kurang signifikan dan dapat direpresentasikan oleh fitur bedrooms dan land area terkait informasinya dan secara korelasinya terhadap harga properti 
                    ''')
        st.markdown('**Multicollinearity Setelah Drop Fitur Floor Area dan Bathrooms**')
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

Multikolinearitas pada variabel independent telah berhasil ditangani dan dikurangi menjadi di bawah 10.0''')
        
    with st.expander('**Kesimpulan EDA**'):
        st.markdown('''
**Kesimpulan EDA**

Berdasarkan EDA (Exploratory Data Analysis) yang telah dilakukan pada dataset properti di Perth, kita dapat mengambil beberapa kesimpulan dan wawasan sebagai berikut:

1. **Jumlah Properti dan Ketersediaan**: Terdapat variasi dalam jumlah properti di berbagai suburb di Perth. Suburb seperti "Bertram," "Iluka," "Bennett Springs," dan "Mindarie" memiliki jumlah properti yang tinggi, mengindikasikan populasi yang padat dan permintaan tinggi untuk tempat tinggal. Namun, ada juga suburb dengan jumlah properti yang lebih sedikit.

2. **Variasi Tipe Properti**: Adanya variasi dalam tipe properti yang tersedia di suburb, yang memengaruhi harga properti. Suburb tertentu mungkin menawarkan jenis properti yang lebih eksklusif atau premium, sementara suburb lainnya memiliki lebih banyak properti yang lebih terjangkau.

3. **Investasi Properti**: Data ini dapat memberikan wawasan bagi pengembang properti atau investor potensial tentang suburb yang mungkin menarik untuk pengembangan lebih lanjut atau investasi properti sesuai dengan target segmentasi investasi mereka (ekslusif atau properti yang lebih terjangkau), melalui insights tanda-tanda pertumbuhan penduduk (dalam data ini direpresentasikan oleh perkembangan pembangunan properti), harga properti, dan juga fasilitas di wilayah sekitar suburb tersebut.

Selain itu, info-info lain mengenai dataset yang berhasil diamati adalah :
- Distribusi data yang tidak merata pada sebagian besar atribut
- Outliers -> adanya indikasi outliers yang signifikan pada sejumlah atribut mengharuskan adanya langkah penanganan outliers lebih lanjut.
- Missing values -> ditemukannya missing values pada beberapa atribut, seperti pada atribut garage, build year, dan nearest school rank juga mengharuskan langkah penanganan missing values lanjutan. Adapun hasil yang didapatkan setelah pengamatan lebih lanjut tentang missing values tersebut juga telah memberikan gambaran langkah yang tepat untuk penanganannya berdasarkan atribut masing-masing, yaitu :
    - Missing values atribut garage : Missing values ini bisa diasumsikan disebabkan oleh adanya beberapa properti yang tidak memiliki garasi (Missing Not At Random) -> Missing values dapat dianggap sebagai "0" atau tidak ada garasi. 
    - Nearest School Distance : Korelasi yang lemah antara atribut ini terhadap harga menunjukan bahwa jarak sekolah terdekat dari properti mungkin bukan faktor yang dipertimbangkan oleh calon pembeli, hal ini bisa disebabkan oleh mungkin minimnya keluarga yang menempati beberapa suburbs sehingga sekolah terdekat bukanlah faktor yang penting, atau adanya kecenderungan calon pembeli untuk menyekolahkan anaknya di sekolah favorit yang sesuai dengan preferensi mereka (tidak bergantung pada dekat atau tidaknya sekolah tersebut dari properti/tempat tinggal mereka). Oleh karena itu, atribut ini akan dieliminasi dari data (didrop)
    - Build Year : Missing values yang didapatkan dari kolom ini diasumsikan sebagai tidak adanya pendataan atau informasi tentang tahun pembangunan properti tersebut. Oleh karena itu, missing values ini juga diasumsikan sebagai nilai not at random. Namun, mengingat korelasinya yang sangat minim terhadap harga properti, penanganan missing values yang akan dilakukan terhadap atribut ini adalah dengan mengeliminasi kolom ini dari data (didrop.)                     ''')