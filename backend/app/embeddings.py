from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(text1, text2):

    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])

    similarity = cosine_similarity(embeddings1, embeddings2)

    return float(similarity[0][0])
    
def contradiction_score(statement, previous_statement):

    similarity = compute_similarity(statement, previous_statement)

    contradiction = 1 -similarity
    return contradiction