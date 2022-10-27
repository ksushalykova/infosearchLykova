import json
import jsonlines

import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

# импорт нужных файлов (следующие 2 только если матрицы и векторизаторы еще не сохранены

import indexation_tfidf
import indexation_bm25

import preprocess


# локальные пути до файлов

f_path = "C:\\Users\marga\Downloads\data.jsonl"
tfidf_matrix_path = "C:\\Users\marga\Downloads\\tfidf_matrix_path.pkl"
tfidf_vectorizer_path = "C:\\Users\marga\Downloads\\tfidf_vectorizer_path.pkl"
bm25_matrix_path = "C:\\Users\marga\Downloads\\bm25_matrix_path.pkl"
count_vectorizer_path = "C:\\Users\marga\Downloads\count_vectorizer_path.pkl"
bert_matrix_path = "C:\\Users\marga\Downloads\\berted_corp.pt"


# функция получения ответов с самым высоким рейтингом автора

def get_answers(f_path):

    answers = []

    with jsonlines.open(f_path, 'r') as f:
        for lines in f:
            ans = lines.get('answers')

            a = []

            for i in ans:
                if len(str(i['author_rating']['value'])) != 0:
                    i['author_rating']['value'] = int(i['author_rating']['value'])
                    a.append(i)
            a.sort(key=lambda x: x['author_rating']['value'], reverse=True)

            if len(a) != 0 and len(a[0]) != 0:

                answers.append(a[0]['text'])
                if len(answers) >= 52000:
                    break

    return answers

# фиксируем список ответов

answers = get_answers(f_path)


# загрузка сохраненных векторайзеров и матриц корпуса (закомменченный код -- сохранение, если еще не сохранено)
# загрузка токенизатора и модели берта (матрица корпуса, индексированного бертом, получена в гугл коллабе и сюда только грузится

def get_tfidf(tfidf_matrix_path, tfidf_vectorizer_path):  # загрузка матрицы и векторайзера для tf-idf
    with open(tfidf_matrix_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(tfidf_vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_matrix, tfidf_vectorizer


def get_bm25(bm25_matrix_path, count_vectorizer_path):  # загрузка матрицы и векторайзера для bm-25
    with open(bm25_matrix_path, 'rb') as f:
        bm25_matrix = pickle.load(f)
    with open(count_vectorizer_path, 'rb') as f:
        count_vectorizer = pickle.load(f)
    return bm25_matrix, count_vectorizer


def get_bert_matrix(bert_matrix_path):   # загрузка матрицы для bert
    corpus_embeddings = torch.load(bert_matrix_path, map_location=torch.device('cpu'))
    bert_corpus_vectors = []
    for i in corpus_embeddings:
        i = i.detach().numpy()
        bert_corpus_vectors.append(i)

    bert_matrix = np.array(bert_corpus_vectors)
    return bert_matrix


def load_bert_model():  # загрузка токенизатора и модели bert
    bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return bert_tokenizer, bert_model


# фиксируем обработанный корпус
corp_prep = preprocess.corp_preprocessed(answers)


# дальше код для случая, если модели и матрицы еще не сохранены

#tfidf_matrix, tfidf_vectorizer = indexation_tfidf.tfidf_matrix_corpus(corp_prep)

#with open(tfidf_matrix_path, 'wb') as f:  # tf-idf матрица сохраняется
#            pickle.dump(tfidf_matrix, f)

#with open(tfidf_vectorizer_path, 'wb') as f:  # tf-idf векторайзер сохраняется
#            pickle.dump(tfidf_vectorizer, f)

#count_vectorizer, bm25_matrix = indexation_bm25.bm_25_matrix_corpus(corp_prep)

#with open(bm25_matrix_path, 'wb') as f:  # bm-25 матрица сохраняется
#            pickle.dump(bm25_matrix, f)

#with open(count_vectorizer_path, 'wb') as f:  # bm-25 векторайзер сохраняется
#            pickle.dump(count_vectorizer, f)


# фиксируем загруженное

tfidf_matrix, tfidf_vectorizer = get_tfidf(tfidf_matrix_path, tfidf_vectorizer_path)
bm25_matrix, count_vectorizer = get_bm25(bm25_matrix_path, count_vectorizer_path)
bert_matrix = get_bert_matrix(bert_matrix_path)
bert_tokenizer, bert_model = load_bert_model()