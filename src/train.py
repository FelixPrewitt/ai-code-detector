from data_loader import load_dataset
from preprocessing import clean_code
from features import extract_features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def train():
    code, labels = load_dataset("github_ai")
    code = clean_code(code)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        code,
        labels,
        test_size=0.2,
        random_state=42, 
    )

    x_train, vectorizer = extract_features(x_train_text)
    

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")

    print("Model and vectorizer saved to /models")

if __name__ == "__main__":
    train()


