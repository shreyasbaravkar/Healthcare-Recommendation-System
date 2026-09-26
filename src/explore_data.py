import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/ to project root
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))

symptom_cols = [col for col in df.columns if "Symptom" in col]
for col in symptom_cols:
    df[col] = df[col].str.strip()
df["Disease"] = df["Disease"].str.strip()

df["Symptoms_List"] = df[symptom_cols].values.tolist()
df["Symptoms_List"] = df["Symptoms_List"].apply(lambda x: [s for s in x if pd.notna(s)])

all_symptoms = sorted(set(s for row in df["Symptoms_List"] for s in row))
print("Total unique symptoms:", len(all_symptoms))

# Load severity weights
severity_df = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
severity_df["Symptom"] = severity_df["Symptom"].str.strip()
severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# Build plain 0/1 checklist
encoded_rows = []
for symptoms in df["Symptoms_List"]:
    row = {symptom: 1 if symptom in symptoms else 0 for symptom in all_symptoms}
    encoded_rows.append(row)

encoded_df = pd.DataFrame(encoded_rows)

# NEW: total severity score per patient (sum of weights of symptoms they have)
encoded_df["Total_Severity"] = encoded_df[all_symptoms].sum(axis=1)

encoded_df["Disease"] = df["Disease"]

print(encoded_df.shape)
print(encoded_df.head())

before = len(encoded_df)
encoded_df = encoded_df.drop_duplicates()
after = len(encoded_df)
print(f"Dropped {before - after} duplicate rows -> {after} unique rows remain")

encoded_df.to_csv(os.path.join(DATA_DIR, "encoded_dataset.csv"), index=False)
print("Saved encoded_dataset.csv (with Total_Severity)")