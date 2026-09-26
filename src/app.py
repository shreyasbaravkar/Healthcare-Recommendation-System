import os
import pandas as pd
import streamlit as st
from recommend import predict_disease, get_recommendation, explain_prediction, symptom_columns

# Bump up font sizes app-wide (Streamlit's defaults are quite small)
st.markdown("""
    <style>
    html, body, [class*="css"] { font-size: 18px; }
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.5rem !important; }
    .stDataFrame { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predict", "Model Insights"])

with tab1:
    st.title("🩺 Personalized Healthcare & Precaution Recommendation")
    st.write("Select your symptoms below to get a prediction, precautions, and an explanation.")

    selected_symptoms = st.multiselect("Select your symptoms:", symptom_columns)

    if st.button("Predict"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            top_diseases = predict_disease(selected_symptoms)

            st.subheader("Top Predictions")
            pred_df = pd.DataFrame(top_diseases, columns=["Disease", "Probability"])
            pred_df["Probability"] = (pred_df["Probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            best_disease = top_diseases[0][0]
            description, precautions = get_recommendation(best_disease)
            explanation = explain_prediction(selected_symptoms, best_disease)

            st.subheader("Description")
            st.write(description)

            st.subheader("Precautions")
            for p in precautions:
                st.write(f"- {p}")

            st.subheader("Why this prediction")
            exp_df = pd.DataFrame(explanation, columns=["Symptom", "Impact"])
            exp_df["Impact"] = exp_df["Impact"].round(3)
            st.dataframe(exp_df, use_container_width=True, hide_index=True)

with tab2:
    st.header("Model Performance (5-fold Cross-Validated)")

    report_df = pd.read_csv(os.path.join("reports", "classification_report.csv"), index_col=0)

    # Pull out the overall accuracy row and show it as a headline number
    accuracy = report_df.loc["accuracy", "f1-score"] if "accuracy" in report_df.index else None
    if accuracy is not None:
        st.metric("Overall Accuracy", f"{accuracy*100:.1f}%")

    st.subheader("Per-Disease Precision, Recall, F1")
    # Drop the accuracy/macro/weighted summary rows from the per-disease table, show those separately
    disease_rows = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    st.dataframe(disease_rows, use_container_width=True)

    st.subheader("Confusion Matrix")
    st.image(os.path.join("reports", "confusion_matrix.png"), caption="Confusion Matrix")