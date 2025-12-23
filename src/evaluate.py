from data_loader import load_dataset
from preprocessing import clean_code
from features import extract_features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate():
    code, labels = load_dataset("basic")
    code = clean_code(code)

    x, _= extract_features(code)
    

    model = LogisticRegression(max_iter=1000)
    model.fit(x, labels)
    
    predictions = model.predict(x)
    acc = accuracy_score(labels, predictions)

    print("Accuracy:", acc)

if __name__ == "__main__":
    evaluate()






