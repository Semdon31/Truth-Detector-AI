from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_opposite(sentence):

    replacements = {
        "never": "",
        "did not": "",
        "didn't": "",
        "not": "",
        "no": "",
    }

    opposite = sentence.lower()

    for word in replacements:
        opposite = opposite.replace(word, "")
    return opposite.strip()

def contradiction_score(sentence):

    opposite = generate_opposite(sentence)

    emb1 = model.encode([sentence])
    emb2 = model.encode([opposite])

    similarity = cosine_similarity(emb1, emb2)[0][0]

    return float(similarity), opposite