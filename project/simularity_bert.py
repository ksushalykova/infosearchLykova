from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# подсчет вектора косинусной близости

def simularity_bert(bert_matrix, query_vec_bert):
    simularity_vec_bert = cosine_similarity(bert_matrix, query_vec_bert)
    simularity_vec = np.reshape(simularity_vec_bert, -1)
    return simularity_vec