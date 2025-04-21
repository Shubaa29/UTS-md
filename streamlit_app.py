import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
try:
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(
        f"❌ Gagal memuat model. Pastikan 'best_model.pkl' adalah hasil pickle model XGBoost. Error detail: {e}"
    )
    st.stop()

# Judul
st.title("Prediksi Kelulusan Mahasiswa")

# Input user
st.header("Masukkan Data Mahasiswa")

ips = st.slider("Rata-rata IPS", 0.0, 4.0, 2.75, 0.01)
sks = st.slider("Jumlah SKS yang sudah diambil", 0, 160, 100)
masa_studi = st.slider("Masa Studi (semester)", 0, 14, 8)
usia = st.slider("Usia Mahasiswa", 17, 30, 21)

# Inputan kategorik
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
pekerjaan_ortu = st.selectbox("Pekerjaan Orang Tua", ["PNS", "Swasta", "Wiraswasta", "Lainnya"])
pendidikan_ortu = st.selectbox("Pendidikan Orang Tua", ["SD", "SMP", "SMA", "D3", "S1", "S2"]) 

# Proses one-hot encoding manual
data = {
    'IPS': ips,
    'SKS': sks,
    'Masa_Studi': masa_studi,
    'Usia': usia,
    'Jenis_Kelamin_Laki-laki': 1 if jenis_kelamin == "Laki-laki" else 0,
    'Jenis_Kelamin_Perempuan': 1 if jenis_kelamin == "Perempuan" else 0,
    'Pekerjaan_Ortu_Lainnya': 1 if pekerjaan_ortu == "Lainnya" else 0,
    'Pekerjaan_Ortu_PNS': 1 if pekerjaan_ortu == "PNS" else 0,
    'Pekerjaan_Ortu_Swasta': 1 if pekerjaan_ortu == "Swasta" else 0,
    'Pekerjaan_Ortu_Wiraswasta': 1 if pekerjaan_ortu == "Wiraswasta" else 0,
    'Pendidikan_Ortu_D3': 1 if pendidikan_ortu == "D3" else 0,
    'Pendidikan_Ortu_S1': 1 if pendidikan_ortu == "S1" else 0,
    'Pendidikan_Ortu_S2': 1 if pendidikan_ortu == "S2" else 0,
    'Pendidikan_Ortu_SD': 1 if pendidikan_ortu == "SD" else 0,
    'Pendidikan_Ortu_SMA': 1 if pendidikan_ortu == "SMA" else 0,
    'Pendidikan_Ortu_SMP': 1 if pendidikan_ortu == "SMP" else 0
}

input_df = pd.DataFrame([data])

# Prediksi
if st.button("Prediksi Kelulusan"):
    pred = model.predict(input_df)[0]
    hasil = "Lulus Tepat Waktu" if pred == 1 else "Tidak Lulus Tepat Waktu"
    st.success(f"Hasil Prediksi: {hasil}")
