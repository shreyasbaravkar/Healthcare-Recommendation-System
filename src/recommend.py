import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/ to project root
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load the saved model and symptom columns
model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
symptom_columns = joblib.load(os.path.join(MODEL_DIR, "symptom_columns.pkl"))

severity_df = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
severity_df["Symptom"] = severity_df["Symptom"].str.strip()
severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# Load lookup tables
desc_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
precaution_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))

desc_df["Disease"] = desc_df["Disease"].str.strip()
precaution_df["Disease"] = precaution_df["Disease"].str.strip()


def predict_disease(user_symptoms, top_n=3):
    # Build the same "checklist" format the model was trained on
    input_data = {symptom: severity_map.get(symptom, 1) if symptom in user_symptoms else 0 for symptom in symptom_columns}
    input_df = pd.DataFrame([input_data])
    input_df["Total_Severity"] = sum(severity_map.get(s, 1) for s in user_symptoms)

    # Get probability for every possible disease, not just the top guess
    probabilities = model.predict_proba(input_df)[0]
    disease_classes = model.classes_

    # Pair each disease with its probability, sort highest first
    results = sorted(zip(disease_classes, probabilities), key=lambda x: x[1], reverse=True)

    top_results = results[:top_n]
    return top_results  # e.g. [("Malaria", 0.8), ("Dengue", 0.15), ("Typhoid", 0.05)]


def get_recommendation(disease):
    description = desc_df[desc_df["Disease"] == disease]["Description"].values
    description = description[0] if len(description) > 0 else "No description found."

    precaution_row = precaution_df[precaution_df["Disease"] == disease]
    if len(precaution_row) > 0:
        precautions = precaution_row.iloc[0, 1:].dropna().tolist()
    else:
        precautions = []

    return description, precautions

import shap

# Build the SHAP explainer once, using the trained model + the full feature set it was trained on
explainer = shap.TreeExplainer(model)

def explain_prediction(user_symptoms, predicted_disease, top_n=3):
    # Rebuild the same "checklist" row used for prediction
    input_data = {symptom: severity_map.get(symptom, 1) if symptom in user_symptoms else 0 for symptom in symptom_columns}
    input_df = pd.DataFrame([input_data])
    input_df["Total_Severity"] = sum(severity_map.get(s, 1) for s in user_symptoms)

    shap_output = explainer.shap_values(input_df)

    # SHAP gives one set of values per disease class — find which "slot" belongs to our predicted disease
    class_index = list(model.classes_).index(predicted_disease)

    # Handle both SHAP output formats depending on installed version
    if isinstance(shap_output, list):
        shap_values_for_disease = shap_output[class_index][0]
    else:
        shap_values_for_disease = shap_output[0, :, class_index]

    feature_names = symptom_columns + ["Total_Severity"]

    # Only show symptoms the user actually reported, ranked by SHAP impact on THIS prediction
    user_feature_importance = [
        (feature_names[i], shap_values_for_disease[i])
        for i in range(len(feature_names))
        if feature_names[i] in user_symptoms
    ]
    user_feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    return user_feature_importance[:top_n]



# Quick test
if __name__ == "__main__":
    test_symptoms = ["fatigue", "vomiting"]
    top_diseases = predict_disease(test_symptoms)

    print("Top predictions:")
    for disease, prob in top_diseases:
        print(f"  {disease}: {prob*100:.1f}%")

    best_disease = top_diseases[0][0]
    description, precautions = get_recommendation(best_disease)
    explanation = explain_prediction(test_symptoms, best_disease)

    print("\nDescription:", description)
    print("Precautions:", precautions)
    print("\nWhy this prediction:")
    for symptom, importance in explanation:
        print(f"  {symptom}: importance {importance:.3f}")