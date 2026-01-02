# AI Code Detector

## Detecting AI-Generated vs Human-Written Code

**Author:** Felix Prewitt

---

## Overview

The AI Code Detector is a personal machine learning project that explores whether a model can distinguish AI-generated code from human-written code using only the source code itself.

The project emphasizes machine learning fundamentals—data preprocessing, feature engineering, experimentation, and evaluation—rather than complex models or deployment infrastructure.

---

## What This Project Does

- Trains a machine learning model on labeled Python code snippets
- Extracts character-level TF-IDF features from source code
- Augments text features with structural features that capture how code is written
- Evaluates authorship predictions on unseen code snippets
- Provides an interactive interface to test predictions on custom code

---

## Tech Stack

- Python
- scikit-learn
- TF-IDF (character-level n-grams)
- Logistic Regression
- Flask (minimal web frontend)
- VS Code
- Git & GitHub

---

## Model and Features

The final model uses a Logistic Regression classifier trained on a combination of:

**Textual Features**
- Character-level TF-IDF (3–5 n-grams)

**Structural Features**
- Number of lines
- Number of loops
- Number of conditional statements
- Number of return statements
- Built-in function usage ratio

These structural features significantly improved performance by capturing how code is constructed, not just which tokens appear.

---

## Results

- Final test accuracy: ~89%
- Structural features resolved many ambiguous cases that TF-IDF alone could not
- The model performs well on:
  - Process-heavy human-written code
  - Concise, idiomatic AI-generated solutions

---

## Limitations

- The model operates on isolated code snippets without execution context or metadata
- Short utility functions may be inherently ambiguous regardless of authorship
- Defensive or verbose AI-generated code can resemble human-written logic
- Accuracy reflects real ambiguity rather than implementation flaws

--

## Status

Backend machine learning pipeline and feature engineering are currently implemented.
Frontend and deployment work is planned next.

