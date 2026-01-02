from data_loader import load_dataset
from preprocessing import clean_code
from features import extract_features
from structure_features import extract_structure_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

import joblib
from pathlib import Path


MODEL_DIR = Path("models")

def evaluate():
    # Load trained artifacts
    model = joblib.load(MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")

    # Load and clean dataset
    code, labels = load_dataset("github_ai")
    code = clean_code(code)

    # Same split as training (important!)
    _, x_test_text, _, y_test = train_test_split(
        code,
        labels,
        test_size=0.2,
        random_state=42,
    )


    # Text features using TRAINED vectorizer
    x_test_text_vec, _ = extract_features(
        x_test_text,
        vectorizer=vectorizer
    )

    # Structural features
    x_test_struct = extract_structure_features(x_test_text)

    # Combine both
    x_test = hstack([x_test_text_vec, x_test_struct])

    # ---- PREDICTION ----
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy}\n")

    # ---- INSPECT PREDICTIONS ----
    print("=== TEST SET PREDICTIONS ===\n")
    for snippet, true, pred in zip(x_test_text, y_test, predictions):
        print("code snippet:")
        print(snippet)
        print()
        print("True label:", "AI" if true == 1 else "HUMAN")
        print("Predicted label:", "AI" if pred == 1 else "HUMAN")
        print("-" * 40)


if __name__ == "__main__":
    evaluate()
