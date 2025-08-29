from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_response(query, response):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, response])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity, 2)
