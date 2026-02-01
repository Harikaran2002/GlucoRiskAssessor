import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Early Stage Diabetes Prediction")

# --------------------------------------------------
# Model selection
# --------------------------------------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
)

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl"
}


# --------------------------------------------------
# Load selected model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model, scaler, feature_cols = load_model(MODEL_PATHS[model_choice])

st.write(f"### Using {model_choice} Model")

# --------------------------------------------------
# Input mode selection
# --------------------------------------------------
input_mode = st.radio(
    "Select Input Method",
    ["Manual Entry", "Upload CSV"]
)

# ================= MANUAL ENTRY ================= #
if input_mode == "Manual Entry":
    st.subheader("Enter Patient Details")

    age = st.number_input("Age", 1, 120, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    polyuria = st.selectbox("Polyuria", [0, 1])
    polydipsia = st.selectbox("Polydipsia", [0, 1])
    sudden_weight_loss = st.selectbox("Sudden Weight Loss", [0, 1])
    weakness = st.selectbox("Weakness", [0, 1])
    polyphagia = st.selectbox("Polyphagia", [0, 1])
    genital_thrush = st.selectbox("Genital Thrush", [0, 1])
    visual_blurring = st.selectbox("Visual Blurring", [0, 1])
    itching = st.selectbox("Itching", [0, 1])
    irritability = st.selectbox("Irritability", [0, 1])
    delayed_healing = st.selectbox("Delayed Healing", [0, 1])
    partial_paresis = st.selectbox("Partial Paresis", [0, 1])
    muscle_stiffness = st.selectbox("Muscle Stiffness", [0, 1])
    alopecia = st.selectbox("Alopecia", [0, 1])
    obesity = st.selectbox("Obesity", [0, 1])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "polyuria": polyuria,
            "polydipsia": polydipsia,
            "sudden_weight_loss": sudden_weight_loss,
            "weakness": weakness,
            "polyphagia": polyphagia,
            "genital_thrush": genital_thrush,
            "visual_blurring": visual_blurring,
            "itching": itching,
            "irritability": irritability,
            "delayed_healing": delayed_healing,
            "partial_paresis": partial_paresis,
            "muscle_stiffness": muscle_stiffness,
            "alopecia": alopecia,
            "obesity": obesity
        }])

        input_data = pd.get_dummies(input_data, drop_first=True)
        input_data = input_data.reindex(columns=feature_cols, fill_value=0)
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

        st.info(f"Probability of Diabetes: {prob:.2f}")

# ================= CSV UPLOAD ================= #
else:
    uploaded_file = st.file_uploader(
        "Upload Test CSV File (semicolon separated)",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=";")

        st.subheader("Uploaded Data Preview")
        st.dataframe(df)

        if "class" in df.columns:
            y_true = df["class"]
            X = df.drop("class", axis=1)
        else:
            X = df
            y_true = None

        X = pd.get_dummies(X, drop_first=True)
        X = X.reindex(columns=feature_cols, fill_value=0)
        X_scaled = scaler.transform(X)

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        df["Prediction"] = y_pred
        df["Probability"] = y_prob

        st.subheader("Prediction Results")
        st.dataframe(df)

        if y_true is not None:
            st.subheader("Classification Report")
            st.text(classification_report(y_true, y_pred))

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
