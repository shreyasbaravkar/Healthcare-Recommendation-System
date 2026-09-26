import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/ to project root
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load data
encoded_df = pd.read_csv(os.path.join(DATA_DIR, "encoded_dataset.csv"))

X = encoded_df.drop("Disease", axis=1)
y = encoded_df["Disease"]

# XGBoost needs numeric labels, not disease names
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5-fold stratified CV: 5 different train/test splits, averaged — more trustworthy than one split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)

rf_scores = cross_val_score(rf, X, y_encoded, cv=cv, scoring="accuracy")
xgb_scores = cross_val_score(xgb, X, y_encoded, cv=cv, scoring="accuracy")

print("Random Forest CV scores:", rf_scores)
print(f"Random Forest mean accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
print()
print("XGBoost CV scores:", xgb_scores)
print(f"XGBoost mean accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")
print()

# Pick whichever model scored higher on average, then retrain it on ALL the data
# (CV was just for evaluation — the final saved model should learn from everything we have)
if rf_scores.mean() >= xgb_scores.mean():
    print("Winner: Random Forest -> training final model on full data")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X, y)  # Random Forest keeps text labels, no encoding needed
    joblib.dump(final_model, os.path.join(MODEL_DIR, "disease_model.pkl"))
    joblib.dump(None, os.path.join(MODEL_DIR, "label_encoder.pkl"))  # placeholder, not used for RF
else:
    print("Winner: XGBoost -> training final model on full data")
    final_model = XGBClassifier(eval_metric="mlogloss", random_state=42)
    final_model.fit(X, y_encoded)  # XGBoost needs the numeric labels
    joblib.dump(final_model, os.path.join(MODEL_DIR, "disease_model.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))  # needed to turn numbers back into disease names

joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "symptom_columns.pkl"))
print("Model and symptom columns saved")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Get honest, cross-validated predictions for every row (no data leakage)
# We use the winning model type here - Random Forest, based on our earlier comparison
cv_predictions = cross_val_predict(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=cv
)

# Confusion matrix - shows which diseases the model confuses with which
fig, ax = plt.subplots(figsize=(16,16))
disp = ConfusionMatrixDisplay.from_predictions(
    y, cv_predictions, xticks_rotation=90, ax=ax, colorbar=False
)
plt.title("Confusion Matrix ( 5-fold Cross-Validated)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Per-disease precision/recall/F1 - shows exactly where the model is weak , not just one overall number 
report_dict = classification_report(y, cv_predictions, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)
report_df.to_csv(os.path.join(REPORTS_DIR, "classification_report.csv"))

