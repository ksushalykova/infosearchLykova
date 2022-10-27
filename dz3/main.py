import numpy as np
import pickle
import argparse
from pathlib import Path

import get_data
import preprocess
import indexation_bm25
import simularity_bm25


def main(query):
    count_vectorizer_path = Path(args.count_vectorizer_path)  # пути как аргументы
    bm25_matrix_path = Path(args.bm25_matrix_path)
    answers = get_data.get_answers(f_path=args.f_path)  # загрузка данных

    # в первый запуск программы раскомменчиваем следующие строки, затем векторайзер и матрица уже будут в виде скачанных файлов, и мы их просто читаем

    # corp_prep = preprocess.corp_preprocessed(answers)
    # count_vectorizer, bm25_matrix = indexation_bm25.bm_25_matrix_corpus(corp_prep)

    # with open(bm25_matrix_path, 'wb') as f:  # матрица сохраняется
    #    pickle.dump(bm25_matrix, f)

    # with open(count_vectorizer_path, 'wb') as f:  # векторайзер сохраняется
    #    pickle.dump(count_vectorizer, f)

    # в остальных случаях просто загружаем векторайзер и матрицу

    with open(count_vectorizer_path, 'rb') as f:
        count_vectorizer = pickle.load(f)

    with open(bm25_matrix_path, 'rb') as f:
        bm25_matrix = pickle.load(f)


    query_prep = preprocess.query_preprocessed(query)  # препроцессинг запроса
    simularity_vec = simularity_bm25.simularity_bm25(bm25_matrix, query_prep, count_vectorizer)  # косинусная близость матрицы и запроса


    # сортировка индексов результатов в порядке убывания значений
    # файлы сортируются в соответствии с индексами результатов

    sorted_indexes = np.argsort(simularity_vec, axis=0)[::-1]
    answers_names = np.array(answers)
    sorted_answers_names = answers_names[sorted_indexes.ravel()]
    print("Самые близкие к запросу результаты \n", sorted_answers_names)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # чтение путей как аргументов
    argparser.add_argument("f_path", help="File path")
    argparser.add_argument("bm25_matrix_path", help="Matrix saved path")
    argparser.add_argument("count_vectorizer_path", help="Vectorizer saved path")
    # запрос
    argparser.add_argument("query", help="Query input")
    args = argparser.parse_args()
    main(query=args.query)