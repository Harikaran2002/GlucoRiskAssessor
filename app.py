import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🩺 Early Stage Diabetes Risk Prediction</h1>
        <p style="font-size:18px;">
        Machine Learning–based clinical decision support system
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
st.subheader("Model Selection")

model_choice = st.selectbox(
    "Choose a trained prediction model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "XGBoost": "model/xgBoost_model.pkl"
}

# --------------------------------------------------
# Load Model (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model, scaler, feature_cols = load_model(MODEL_PATHS[model_choice])

st.success(f"Using **{model_choice}** model")

st.markdown("---")

# --------------------------------------------------
# Input Method
# --------------------------------------------------
st.subheader("Input Method")

input_mode = st.radio(
    "Select how you want to provide patient data",
    ["Upload CSV (Test Data)", "Manual Entry"],
    horizontal=True
)

# --------------------------------------------------
# CSV UPLOAD (PROFESSIONAL FIRST)
# --------------------------------------------------
if input_mode == "Upload CSV (Test Data)":

    st.subheader("Upload Test Dataset")

    # Sample test data
    sample_df = pd.read_csv("data/diabetes_test.csv", sep=";")

    st.download_button(
        label="⬇Download Sample Test Data",
        data=sample_df.to_csv(index=False, sep=";"),
        file_name="sample_diabetes_test_data.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file (semicolon separated)",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=";")

        st.subheader("Uploaded Data Preview")
        st.dataframe(df, use_container_width=True)

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

        st.markdown("---")
        st.subheader("Prediction Results")
        st.dataframe(df, use_container_width=True)

       # ---------------- METRICS ---------------- #
        st.markdown("---")
        st.subheader("Model Evaluation")

        if "class" not in df.columns:
            st.info(
                "Evaluation metrics and confusion matrix require ground-truth labels. "
                "Please upload a labeled test dataset containing the `class` column."
            )
        else:
            # Ground truth and predictions
            y_true = df["class"].astype(int)
            y_pred = df["Prediction"].astype(int)
            y_prob = df["Probability"]

            # ---------------- ROW 1: METRICS + CONFUSION MATRIX ---------------- #
            left_col, right_col = st.columns([1.3, 1])

            with left_col:
                st.markdown("Evaluation Metrics")

                # Exact order as per assignment
                st.metric("1. Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                st.metric("2. AUC Score", f"{roc_auc_score(y_true, y_prob):.4f}")
                st.metric("3. Precision", f"{precision_score(y_true, y_pred):.4f}")
                st.metric("4. Recall", f"{recall_score(y_true, y_pred):.4f}")
                st.metric("5. F1 Score", f"{f1_score(y_true, y_pred):.4f}")
                st.metric(
                    "6. Matthews Correlation Coefficient (MCC)",
                    f"{matthews_corrcoef(y_true, y_pred):.4f}"
                )

            with right_col:
                st.markdown("Confusion Matrix")

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=ax
                )
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

         # ---------------- CLASSIFICATION REPORT (PROFESSIONAL TABLE) ---------------- #
        st.markdown("---")
        st.subheader("Classification Report")

        # Generate report as dictionary
        report = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose().round(3)

        # Rename index for clarity
        report_df.rename(
            index={
                "0": "Class 0 (Non-Diabetic)",
                "1": "Class 1 (Diabetic)",
                "accuracy": "Overall Accuracy",
                "macro avg": "Macro Average",
                "weighted avg": "Weighted Average"
            },
            inplace=True
        )

        # Display table
        st.dataframe(
            report_df,
            use_container_width=True
        )

# --------------------------------------------------
# MANUAL ENTRY
# --------------------------------------------------
else:
    st.subheader("Manual Patient Data Entry")
    st.info("Manual entry is intended for individual patient prediction.")

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

    if st.button("🔍 Predict Diabetes Risk"):
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

        st.markdown("---")
        if pred == 1:
            st.error("High Risk of Diabetes Detected")
        else:
            st.success("Low Risk of Diabetes")

        st.metric("Probability of Diabetes", f"{prob:.2f}")
