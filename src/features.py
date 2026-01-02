from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(code_list, vectorizer=None):
    if vectorizer is None:
        # Training mode: fit a new vectorizer
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5)
        )
        x = vectorizer.fit_transform(code_list)
        return x, vectorizer
    else:
        # Evaluation mode: reuse existing vectorizer
        x = vectorizer.transform(code_list)
        return x, vectorizer

