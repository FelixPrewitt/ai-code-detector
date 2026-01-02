AI Code Detector 

Detecting AI-Generated vs Human-Written Code

Author: Felix Prewitt

 The AI Code Detector is a personal machine learning project that explores whether models can distinguish AI-generated code from human-written code using only the source code itself.

 The project focuses on machine learning fundamentals—data preprocessing, feature extraction, dataset design, and experimentation—rather than model complexity. It is designed as a learning-first, portfolio project that emphasizes understanding model behavior and real-world limitations.

- What This Project Does

 Trains a machine learning model on labeled Python code snippets
 Analyzes source code using character-level TF-IDF features
 Evaluates how well authorship signals can be learned from isolated code
 Investigates where ambiguity naturally limits detection accuracy

- Tech Stack

    Python
    scikit-learn
    TF-IDF (character-level n-grams)
    VS Code
    Git & GitHub

Local development on macOS (M1).

📂 Project Structure
ai_code_detector/
├── data/
│   └── samples/
│       └── github_ai/
│           ├── code.txt
│           └── labels.txt
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── tests/
└── README.md

 Current Status

    End-to-end ML pipeline implemented
    Dataset includes human-written code, AI-generated code, and paired examples
    Model training and evaluation working as expected   
    Actively iterating and improving dataset quality

 Purpose

  This project demonstrates:
    Practical machine learning fundamentals
    Thoughtful dataset and experiment design
    Clean Python project organization
    Applied problem-solving beyond coursework
    
Future Work

    Expand and refine the dataset
    Improve authorship detection through targeted examples
    Experiment with additional feature representations
    Explore more advanced or hybrid detection approaches

