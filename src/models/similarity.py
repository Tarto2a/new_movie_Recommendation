from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(corpus):
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return similarity_matrix
