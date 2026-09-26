# 🩺 Personalized Healthcare & Precaution Recommendation System

A machine learning system that predicts likely diseases from user-reported symptoms and recommends relevant precautions — built with calibrated confidence, cross-validated evaluation, and per-prediction explainability.

## Problem Statement

Most basic symptom-checker projects map symptoms to a single disease with no sense of confidence, no real evaluation, and no explanation of *why*. This project aims for something closer to a real triage tool: transparent, uncertainty-aware, and rigorously evaluated.

## What Makes This Different

- **Calibrated confidence, not one guess**: Returns the top-3 likely diseases with real probability scores, using a Random Forest chosen after being benchmarked against XGBoost.
- **Rigorous, leak-free evaluation**: The raw dataset contained ~94% duplicate rows (4,616 of 4,921) caused by reordered symptom listings. These were identified and removed before evaluation — without this, accuracy numbers would be artificially inflated by train/test overlap. All reported metrics come from 5-fold stratified cross-validation, not a single lucky split.
- **Severity-aware prediction**: A `Total_Severity` feature (derived from a symptom-severity weighting dataset) captures how serious the overall symptom combination is, not just which symptoms are present.
- **Real explainability**: Uses SHAP (SHapley Additive exPlanations) to show which of the user's specific symptoms pushed the prediction toward — or away from — the predicted disease, per prediction, not just a general "this symptom matters overall" statement.
- **Built-in model insights dashboard**: A dedicated tab shows the confusion matrix and per-disease precision/recall/F1, all computed honestly from cross-validated predictions.

## Tech Stack

- Python, pandas
- scikit-learn (Random Forest, cross-validation)
- XGBoost (benchmarked alternative)
- SHAP (explainability)
- Streamlit (web interface)
- pytest (automated tests)

## Project Structure
healthcare_project/
├── data/ # Raw and processed datasets
├── models/ # Saved trained model, symptom columns
├── reports/ # Generated confusion matrix + classification report
├── src/
│ ├── explore_data.py # Data cleaning, encoding, deduplication
│ ├── train_model.py # Model comparison, training, evaluation, report generation
│ ├── recommend.py # Prediction, recommendation, SHAP explainability
│ └── app.py # Streamlit web app
├── tests/
│ └── test_recommend.py # Automated tests for core logic
└── requirements.txt


## How to Run

```bash
pip install -r requirements.txt

python src/explore_data.py     # one-time: cleans and encodes data
python src/train_model.py      # one-time: compares models, trains, evaluates, saves reports
pytest tests/                  # run automated tests
streamlit run src/app.py       # launch the app
```

## Dataset

Disease-Symptom dataset (Kaggle, itachi9604) — includes symptoms, disease descriptions, precautions, and symptom severity weights.

## Model Evaluation

| Model | 5-fold CV Mean Accuracy |
|---|---|
| Random Forest | 100% (consistent across all folds) |
| XGBoost | 90.5% |

Random Forest was selected. Note: with ~305 unique symptom patterns spread across 41 diseases (roughly 5-10 samples per class), classes are cleanly separable — a known characteristic of this dataset, not a claim of real-world clinical accuracy. This is stated transparently rather than presented as a inflated result.

## Future Improvements

- Natural-language symptom input (free-text, matched to known symptoms) instead of manual selection
- Larger, real-world clinical dataset
- Containerized deployment (Docker) and public hosting