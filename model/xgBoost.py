import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from xgboost import XGBClassifier

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/diabetes_data.csv", sep=";")

X = df.drop("class", axis=1)
y = df["class"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# Feature scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# XGBoost model
# --------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# --------------------------------------------------
# Metrics
# --------------------------------------------------
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

print("XGBoost Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
with open("model/xgBoost_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X_train.columns), f)
