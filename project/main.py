import numpy as np
import streamlit
import os
import time

import get_data
import preprocess
import indexation_tfidf
import query_indexation_bert
import simularity_tfidf
import indexation_bm25
import simularity_bm25
import simularity_bert

from get_data import corp_prep
from get_data import answers
from get_data import tfidf_matrix, tfidf_vectorizer
from get_data import bm25_matrix, count_vectorizer
from get_data import bert_matrix, bert_tokenizer, bert_model


# сортировка индексов результатов в порядке убывания значений
# файлы сортируются в соответствии с индексами результатов

def sorting_results(simularity_vec, answers):
    sorted_indexes = np.argsort(simularity_vec, axis=0)[::-1]
    answers_names = np.array(answers)
    if sorted_indexes[0] == 0:
        sorted_answers_names = []
    else:
        sorted_answers_names = answers_names[sorted_indexes.ravel()]
    return sorted_answers_names

# поиск методом tf-idf

def search_tfidf(query_prep):
    simularity_vec = simularity_tfidf.simularity_tfidf(tfidf_matrix, query_prep, tfidf_vectorizer)
    sorted_results = sorting_results(simularity_vec, answers)
    return sorted_results

# поиск методом bm-25

def search_bm25(query_prep):
    simularity_vec = simularity_bm25.simularity_bm25(bm25_matrix, query_prep, count_vectorizer)
    sorted_results = sorting_results(simularity_vec, answers)
    return sorted_results

# поиск методом bert

def search_bert(query_prep):
    query_vector_bert = query_indexation_bert.query_vec_bert_indexation(bert_tokenizer, bert_model, query_prep)
    simularity_vec = simularity_bert.simularity_bert(bert_matrix, query_vector_bert)
    sorted_results = sorting_results(simularity_vec, answers)
    return sorted_results

# интерфейс - streamlit

if __name__ == '__main__':

    streamlit.title('THREE-WAY BROWSER')
    streamlit.sidebar.title('ПОИСКОВИК ПО КОРПУСУ ОТВЕТОВ MAIL.RU')
    page_bg_img = '''
        <style>
        body {
        background-image: url("https://fikiwiki.com/uploads/posts/2022-02/1644814472_42-fikiwiki-com-p-kartinki-s-krasotoi-56.jpg");
        background-size: cover;
        }
        </style>
        '''
    streamlit.markdown(page_bg_img, unsafe_allow_html=True)
    streamlit.sidebar.info('Поисковик с тремя методами поиска (TF-IDF, BM-25, BERT), в соответствующую строку нужно ввести запрос, в следующей выбрать метод поиска и нажать "ИСКАТЬ". Поисковик выдаст топ-10 самых близких к запросу результатов, а также время поиска).')
    url = "https://fikiwiki.com/uploads/posts/2022-02/1644814472_42-fikiwiki-com-p-kartinki-s-krasotoi-56.jpg"
    streamlit.image(url, caption=url, width=600)

    query = streamlit.text_input('ЗАПРОС:')
    search_way = streamlit.selectbox('МЕТОД ПОИСКА:', ['TF-IDF', 'BM25', 'BERT'])
    streamlit.write(f"Вы выбрали: {search_way!r}")
    start = time.time()
    query_prep = preprocess.query_preprocessed(query)

    if streamlit.button('ИСКАТЬ'):
        if search_way == 'TF-IDF':
            sorted_results = search_tfidf(query_prep)
        elif search_way == 'BM25':
            sorted_results = search_bm25(query_prep)
        elif search_way == 'BERT':
            sorted_results = search_bert(query_prep)
        streamlit.write("САМЫЕ БЛИЗКИЕ РЕЗУЛЬТАТЫ К ЗАПРОСУ:")
        if len(sorted_results) != 0:
            streamlit.table(sorted_results[:10])
            end = time.time()
            streamlit.write("ВРЕМЯ ПОИСКА (В СЕКУНДАХ) \n", end - start)
        else:
            streamlit.write("ПО ЗАПРОСУ НИЧЕГО НЕ НАЙДЕНО")