import pickle

model = pickle.load(open("models/deception_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_deception(text):

    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)
    score = probabilities.max()
    return float(score)