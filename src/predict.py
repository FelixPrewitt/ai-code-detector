import joblib
from pathlib import Path
from preprocessing import clean_code

MODEL_DIR = Path("models")

def predict(code_snippet):
    #Load trained model
    model = joblib.load(MODEL_DIR / "model.joblib" )
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")

    #clean input code
    cleaned = clean_code(code_snippet)

    x = vectorizer.transform(cleaned)

    predictions = model.predict(x)

    return predictions 

if __name__ == "__main__":
    examples = {
        'print("hello world")',
        'def solve(): return [i*i for i in range(10)]'
    }
 
    preds = predict(examples)

    for code, label in zip(examples, preds):
        result = "AI-written" if label == 1 else "Human-written"
        print(f"{result}: {code} ")


