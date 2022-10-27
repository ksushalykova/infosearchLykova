import indexation_bm25

# подсчет вектора косинусной близости

def simularity_bm25(bm25_matrix, query_prep, count_vectorizer):

    query_vec_bm25 = indexation_bm25.bm_25_index_query(query_prep, count_vectorizer)
    simularity_vec_bm25 = bm25_matrix*query_vec_bm25.T
    simularity_vec = simularity_vec_bm25.toarray()

    return simularity_vec