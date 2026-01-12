import joblib
from pathlib import Path

from preprocessing import clean_code
from features import extract_features
from structure_features import extract_structure_features
from scipy.sparse import hstack

MODEL_DIR = Path("models")

def predict_code(code_snippet: str) -> str:
    #load arrtifacts 
    model = joblib.load(MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")

    code_cleaned = clean_code([code_snippet])

    x_text, _ = extract_features(code_cleaned, vectorizer=vectorizer)
    x_struct = extract_structure_features(code_cleaned)

    x = hstack([x_text, x_struct])

    prediction = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]


    return {
        "label": "AI" if prediction == 1 else "HUMAN",
        "confidence": float(max(probabilities))
    }



