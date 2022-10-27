from sklearn.feature_extraction.text import TfidfVectorizer


# получение индексированного корпуса (матрицы) для tf-idf

def tfidf_matrix_corpus(corp_prep):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corp_prep)
    return tfidf_matrix, tfidf_vectorizer

# получение индексированного (вектора) запроса

def tfidf_index_query(query_prep, tfidf_vectorizer):
    query_vec_tfidf = tfidf_vectorizer.transform([query_prep])
    return query_vec_tfidf