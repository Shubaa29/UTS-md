from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Copy dataset
data = df.copy()

# Handle categorical encoding
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership',
                    'loan_intent', 'previous_loan_defaults_on_file']

# Encode categorical features
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Define features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_preds)

# Select best model
best_model = rf_model if rf_acc >= xgb_acc else xgb_model
best_model_name = "Random Forest" if rf_acc >= xgb_acc else "XGBoost"

# Save best model
with open("/mnt/data/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

(best_model_name, rf_acc, xgb_acc)
