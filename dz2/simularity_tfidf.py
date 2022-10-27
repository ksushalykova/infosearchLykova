import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import indexation_tfidf

# подсчет вектора косинусной близости

def simularity_tfidf(tfidf_matrix, query_prep, tfidf_vectorizer):
    query_vec_tfidf = indexation_tfidf.tfidf_index_query(query_prep, tfidf_vectorizer)
    simularity_vec_tfidf = cosine_similarity(tfidf_matrix, query_vec_tfidf)
    simularity_vec = np.reshape(simularity_vec_tfidf, -1)
    return simularity_vec