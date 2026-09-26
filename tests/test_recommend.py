import sys
import os

# Let this test file find and import from src/, since tests/ is a separate folder 
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","src"))

from recommend import predict_disease, get_recommendation, explain_prediction

def test_predict_disease_returns_three_results():
    results = predict_disease(["fatigue"])
    assert len(results) == 3

def test_predict_disease_probabilities_are_valid():
    results = predict_disease(["fatigue"])
    for disease , prob in results:
        assert 0 <= prob <=1 

def test_get_recommendation_returns_description_and_precautions():
    description, precautions = get_recommendation("GERD")
    assert isinstance(description, str) and len(description) > 0
    assert isinstance(precautions, list) and len(precautions) > 0

def test_explain_prediction_only_uses_selected_symptoms():
    symptoms = ["fatigue","vomiting"]
    explanation = explain_prediction(symptoms, "GERD")
    for symptom, impact in explanation:
        assert symptom in symptoms
        