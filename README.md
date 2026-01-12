# AI Code Detector

**Detecting AI-Generated vs Human-Written Code**
**Author:** Felix Prewitt

---

## Overview

AI Code Detector is a machine learning project designed to classify **human-written code vs AI-generated code**. The goal of this project is to explore **stylistic, structural, and statistical differences** between human and AI code using classical machine learning techniques and transformer-based embeddings.

This project is intentionally built to be **local-machine friendly**, **reproducible**, and **educational**, serving as a foundation for future **LLM fine-tuning and research work**.

---

## What This Project Does

* Trains a machine learning model on labeled Python code snippets
* Extracts **character-level TF-IDF features** from source code
* Augments text features with **structural features** that capture how code is written
* Evaluates authorship predictions on unseen code snippets
* Provides an **interactive interface** to test predictions on custom code

---

## Tech Stack

* Python
* scikit-learn
* TF-IDF (character-level n-grams)
* Logistic Regression
* Flask (minimal web frontend)
* VS Code
* Git & GitHub

---

## Model and Features

The final model uses a **Logistic Regression classifier** trained on a combination of textual and structural features.

### Textual Features

* Character-level TF-IDF (3–5 n-grams)

### Structural Features

* Number of lines
* Number of loops
* Number of conditional statements
* Number of return statements
* Built-in function usage ratio

These structural features significantly improved performance by capturing **how code is constructed**, not just which tokens appear.

---

## Results

* **Final test accuracy:** ~89%
* Structural features resolved many ambiguous cases that TF-IDF alone could not

The model performs well on:

* Process-heavy human-written code
* Concise, idiomatic AI-generated solutions

---

## Limitations

* Operates on isolated code snippets without execution context or metadata
* Short utility functions may be inherently ambiguous regardless of authorship
* Defensive or verbose AI-generated code can resemble human-written logic

Accuracy reflects **real ambiguity**, not implementation flaws.

---

## Status

The backend machine learning pipeline and feature engineering are currently implemented. Frontend improvements and deployment work are planned next.

---

## Author

**Felix Prewitt**

---

## License

Educational and research use only.

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-code-detector.git
cd ai-code-detector
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\\Scripts\\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

To start the local Flask interface:

```bash
python src/app.py
```

Once running, open your browser and navigate to:

```
http://127.0.0.1:5000
```

You can paste Python code snippets into the interface to receive an **AI vs Human authorship prediction**.
