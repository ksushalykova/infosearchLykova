import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse


# получение индексированного корпуса (матрицы) для bm-25

def bm_25_matrix_corpus(corp_prep):
    k = 2
    b = 0.75

    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(corp_prep)
    tf = count

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(corp_prep)

    idf = tfidf_vectorizer.idf_

    len_d = np.array(count.sum(axis=1)).ravel()

    avdl = count.sum(axis=1).mean()

    rows = []
    cols = []
    scores = []

    for i, j in zip(*tf.nonzero()):
        cur_tf = tf[i, j]
        cur_idf = idf[j]
        cur_len_d = len_d[i]

        A = cur_idf * cur_tf * (k + 1)
        B_1 = k * (1 - b + b * (cur_len_d / avdl))
        B = cur_tf + B_1
        cur_score = A / B

        rows.append(i)
        cols.append(j)
        scores.append(cur_score)

    bm25_matrix = sparse.csr_matrix((scores, (rows, cols)))

    return count_vectorizer, bm25_matrix


# получение индексированного (вектора) запроса

def bm_25_index_query(query, count_vectorizer):
    query_vec_bm25 = count_vectorizer.transform([query])

    return query_vec_bm25