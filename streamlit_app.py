import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Path ke model
MODEL_PATH = "best_model.pkl"
# Cek keberadaan file model sebelum loading
i
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ File '{MODEL_PATH}' tidak ditemukan. Pastikan file ini di-upload ke GitHub sejajar dengan streamlit_app.py.")
    st.stop()

# Load trained model dengan penanganan error
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Inisialisasi scaler (hanya placeholder, sebaiknya load scaler nyata dari training)
scaler = StandardScaler()

def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    # Dummy encoding
    df = pd.get_dummies(df)
    # Kolom yang diharapkan model
    expected_columns = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
        'person_gender_male', 'person_education_High School', 'person_education_Master',
        'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
        'loan_intent_VENTURE', 'previous_loan_defaults_on_file_Yes'
    ]
    for col in expected_columns:
        if col not in df:
            df[col] = 0
    df = df[expected_columns]
    # Skala data
    return scaler.fit_transform(df)

# Judul aplikasi
title = "Loan Approval Prediction App"
st.title(title)
st.write("Masukkan detail pemohon di bawah ini untuk memprediksi persetujuan pinjaman.")

# Form input user
with st.form("loan_form"):
    age = st.number_input("Usia", 18, 100, 30)
    gender = st.selectbox("Jenis Kelamin", ["male", "female"])
    education = st.selectbox("Pendidikan", ["Bachelor", "High School", "Master"])
    income = st.number_input("Pendapatan", 1000, 1000000, 50000)
    emp_exp = st.number_input("Lama Kerja (tahun)", 0, 50, 5)
    ownership = st.selectbox("Status Tempat Tinggal", ["RENT", "OWN", "MORTGAGE"])
    loan_amt = st.number_input("Jumlah Pinjaman", 1000, 50000, 15000)
    intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    int_rate = st.slider("Bunga Pinjaman (%)", 5.0, 25.0, 12.5)
    percent_income = st.slider("Persentase Pinjaman terhadap Pendapatan", 0.1, 1.0, 0.3)
    cred_hist = st.number_input("Lama Riwayat Kredit (tahun)", 0, 20, 4)
    score = st.number_input("Skor Kredit", 300, 850, 650)
    prev_default = st.selectbox("Pernah Menunggak Sebelumnya?", ["No", "Yes"])
    submit = st.form_submit_button("Prediksi")

if submit:
    user_input = {
        'person_age': age,
        'person_gender': gender,
        'person_education': education,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': ownership,
        'loan_amnt': loan_amt,
        'loan_intent': intent,
        'loan_int_rate': int_rate,
        'loan_percent_income': percent_income,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': score,
        'previous_loan_defaults_on_file': prev_default
    }
    X_input = preprocess_input(user_input)
    result = model.predict(X_input)[0]
    st.success("Hasil Prediksi: **Disetujui**" if result == 1 else "Hasil Prediksi: **Ditolak**")

# Contoh Test Case
st.markdown("---")
st.subheader("Contoh Test Case")
st.markdown(
"""
**Case 1:**
- Usia: 35
- Gender: male
- Pendidikan: Bachelor
- Pendapatan: 80000
- Lama kerja: 8 tahun
- Status rumah: OWN
- Jumlah pinjaman: 20000
- Tujuan: PERSONAL
- Bunga: 10%
- Persentase pendapatan: 0.25
- Riwayat kredit: 6 tahun
- Skor kredit: 720
- Default sebelumnya: No

**Case 2:**
- Usia: 24
- Gender: female
- Pendidikan: High School
- Pendapatan: 30000
- Lama kerja: 2 tahun
- Status rumah: RENT
- Jumlah pinjaman: 15000
- Tujuan: VENTURE
- Bunga: 20%
- Persentase pendapatan: 0.5
- Riwayat kredit: 2 tahun
- Skor kredit: 580
- Default sebelumnya: Yes
"""
)
