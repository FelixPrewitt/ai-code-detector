from data_loader import load_dataset
from preprocessing import clean_code
from features import extract_features
from sklearn.linear_model import LogisticRegression

def train():
    code, labels = load_dataset("basic")
    code = clean_code(code)

    x, vectorizer = extract_features(code)

    model = LogisticRegression(max_iter=1000)
    model.fit(x, labels)

    print("Model trained successfully!")
    return model, vectorizer

if __name__ == "__main__":
    train()
