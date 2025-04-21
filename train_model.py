import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle

# 1. Load data
df = pd.read_csv("Dataset_A_loan.csv")

# 2. Pisahkan fitur dan target
X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

# (Opsional) encoding jika perlu
# Misalnya: X = pd.get_dummies(X)  <-- jika ada kolom kategori

# 3. Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Buat pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # kalau tidak perlu scaling, bisa dihapus
    ('model', XGBClassifier())
])

# 5. Latih model
pipeline.fit(X_train, y_train)

# 6. Simpan pipeline
with open("best_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
