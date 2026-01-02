from data_loader import load_dataset
from preprocessing import clean_code
from features import extract_features
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = Path("models")


def evaluate():
   #load train model and vectorizer 
    model = joblib.load(MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")

    #load cleaned full dataset
    code, labels = load_dataset("github_ai")
    code = clean_code(code)

    #same splt dataset as in train.py
    _, x_test_text, _, y_test = train_test_split(
        code,
        labels,
        test_size=0.2,
        random_state=42
    )


    #transform test data using loaded vectorizer
    x_test = vectorizer.transform(x_test_text)

    #make predictions
    pred = model.predict(x_test)

    #accuracy
    accuracy = accuracy_score(y_test, pred)
    print(f"\nAccuracy: {accuracy}\n")

    #inspect some predictions 
    label_map = {0: "HUMAN", 1: "AI"}

    print("=== TEST SET PREDICTIONS ===\n")

    for i in range(len(x_test_text)):
        print("code snippet:")
        print(x_test_text[i])
        print()

        print(f"True label: {label_map[y_test[i]]}")
        print(f"Predicted label: {label_map[pred[i]]}")
        print("-" * 40)



if __name__ == "__main__":
    evaluate()






