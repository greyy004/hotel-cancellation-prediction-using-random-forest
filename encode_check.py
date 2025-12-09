import pickle

# ===== Load encoders =====
ENCODERS_PATH = "model_files/encoders.pkl"
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

# ===== Print exact LabelEncoder mappings =====
for col, le in encoders.items():
    print(f"\nColumn: {col}")
    for category in le.classes_:
        code = le.transform([category])[0]  # ensures exact encoding used by the model
        print(f"  {category} --> {code}")
