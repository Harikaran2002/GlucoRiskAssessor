import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

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
# (Not required for Random Forest,
#  but kept for pipeline consistency)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Random Forest model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
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

print("Random Forest Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
with open("model/random_forest_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X_train.columns), f)
