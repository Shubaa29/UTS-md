import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model dan scaler
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as f:
    scaler, expected_columns = pickle.load(f)

# Preprocessing input user
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)

    # Tambah kolom yang hilang agar sesuai dengan kolom training
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
    prediction = model.predict(np.array(processed))[0]
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
- Lama kerja: 5 tahun  
- Status rumah: RENT  
- Jumlah pinjaman: 24000  
- Tujuan: PERSONAL  
- Bunga: 12%  
- Rasio pendapatan: 0.3  
- Lama kredit: 4 tahun  
- Skor kredit: 650  
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
