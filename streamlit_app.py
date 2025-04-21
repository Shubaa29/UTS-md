import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)

    expected_columns = ['person_age', 'person_income', 'person_emp_exp',
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                        'cb_person_cred_hist_length', 'credit_score',
                        'person_gender_male', 'person_education_High School',
                        'person_education_Master', 'person_home_ownership_OWN',
                        'person_home_ownership_RENT', 'loan_intent_EDUCATION',
                        'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
                        'loan_intent_VENTURE', 'previous_loan_defaults_on_file_Yes']
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    
    scaled = scaler.transform(df)  
    return scaled

# Streamlit UI
st.title("Aplikasi Prediksi Persetujuan Pinjaman")
st.write("Isi form di bawah ini untuk melihat prediksi kelulusan pinjaman.")

with st.form("form_input"):
    age = st.number_input("Usia", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["male", "female"])
    education = st.selectbox("Pendidikan", ["Bachelor", "High School", "Master"])
    income = st.number_input("Pendapatan", min_value=1000, max_value=1000000, value=50000)
    emp_exp = st.number_input("Lama Kerja (tahun)", 0, 50, 5)
    home = st.selectbox("Status Tempat Tinggal", ["RENT", "OWN", "MORTGAGE"])
    loan_amnt = st.number_input("Jumlah Pinjaman", 1000, 50000, 15000)
    intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    rate = st.slider("Bunga Pinjaman (%)", 5.0, 25.0, 12.5)
    percent_income = st.slider("Rasio Pinjaman vs Pendapatan", 0.1, 1.0, 0.3)
    cred_len = st.number_input("Lama Riwayat Kredit (tahun)", 0, 20, 4)
    score = st.number_input("Skor Kredit", 300, 850, 650)
    default = st.selectbox("Pernah Menunggak?", ["No", "Yes"])
    
    submit = st.form_submit_button("Prediksi")

if submit:
    user_input = {
        'person_age': age,
        'person_gender': gender,
        'person_education': education,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home,
        'loan_amnt': loan_amnt,
        'loan_intent': intent,
        'loan_int_rate': rate,
        'loan_percent_income': percent_income,
        'cb_person_cred_hist_length': cred_len,
        'credit_score': score,
        'previous_loan_defaults_on_file': default
    }

    processed = preprocess_input(user_input)
    prediction = model.predict(processed)[0]
    result = "✅ Disetujui" if prediction == 1 else "❌ Ditolak"
    st.success(f"Hasil Prediksi: {result}")

# Test Case
st.markdown("---")
st.subheader("Contoh Test Case")

st.markdown("""
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
- Rasio pendapatan: 0.25  
- Lama kredit: 6 tahun  
- Skor kredit: 720  
- Pernah menunggak: No  

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
- Rasio pendapatan: 0.5  
- Lama kredit: 2 tahun  
- Skor kredit: 580  
- Pernah menunggak: Yes  
""")
