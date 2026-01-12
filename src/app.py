from flask import Flask, render_template, request
import joblib
from pathlib import Path
import numpy as np

from preprocessing import clean_code
from features import extract_features
from structure_features import extract_structure_features
from scipy.sparse import hstack

app = Flask(__name__)

# Load model + vectorizer
MODEL_DIR = Path("models")
model = joblib.load(MODEL_DIR / "model.joblib")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    code = ""

    if request.method == "POST":
        code = request.form.get("code", "")

        if code.strip():
            cleaned = clean_code([code])

            # Text features
            text_vec = vectorizer.transform(cleaned)

            # Structural features
            struct_vec = extract_structure_features(cleaned)

            # Combine
            x = hstack([text_vec, struct_vec])

            # Predict
            probs = model.predict_proba(x)[0]
            pred = model.predict(x)[0]

            label = "AI" if pred == 1 else "HUMAN"
            confidence = float(np.max(probs))

            result = {
                "label": label,
                "confidence": confidence
            }

    return render_template(
        "index.html",
        result=result,
        code=code
    )


if __name__ == "__main__":
   app.run(host="127.0.0.1", port=8000, debug=False)
