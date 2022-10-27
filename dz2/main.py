import pandas as pd
import pickle
import argparse
from pathlib import Path

import get_data
import preprocess
import indexation_tfidf
import simularity_tfidf


def main(query):
    tfidf_vectorizer_path = Path(args.tfidf_vectorizer_path) # пути как аргументы
    tfidf_matrix_path = Path(args.tfidf_matrix_path)
    texts, f_names, f_paths = get_data.get_texts(f_path=args.f_path)  # загрузка данных

    # в первый запуск программы раскомменчиваем следующие строки, затем векторайзер и матрица уже будут в виде скачанных файлов, и мы их просто читаем

    #corp_prep = preprocess.corp_preprocessed(texts)
    #tfidf_matrix, tfidf_vectorizer = indexation_tfidf.tfidf_matrix_corpus(corp_prep)

    #with open(tfidf_matrix_path, 'wb') as f:  # матрица сохраняется
    #    pickle.dump(tfidf_matrix, f)

    #with open(tfidf_vectorizer_path, 'wb') as f:  # векторайзер сохраняется
    #    pickle.dump(tfidf_vectorizer, f)


    # в остальных случаях просто загружаем векторайзер и матрицу

        with open(tfidf_vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        with open(tfidf_matrix_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)


    query_prep = preprocess.query_preprocessed(query)  # препроцессинг запроса
    simularity_vec = simularity_tfidf.simularity_tfidf(tfidf_matrix, query_prep, tfidf_vectorizer)  # косинусная близость матрицы и запроса

    # словарь и таблица с соответствиями номеров текстов и их названий

    dict = {'simularity_vector': simularity_vec, 'names': f_names}
    table = pd.DataFrame(dict)

    table = table.sort_values(by=['simularity_vector'], ascending=False)
    print("Самые близкие к запросу результаты \n", table['names'])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # чтение путей как аргументов
    argparser.add_argument("f_path", help="File path")
    argparser.add_argument("tfidf_matrix_path", help="Matrix saved path")
    argparser.add_argument("tfidf_vectorizer_path", help="Vectorizer saved path")
    # запрос
    argparser.add_argument("query", help="Query input")
    main(query=args.query)