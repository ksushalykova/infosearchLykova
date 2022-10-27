import argparse
import numpy as np

from pathlib import Path

import get_data
import preprocess
import indexation_bert
import simularity_bert
from indexation_bert import bert_tokenizer, bert_model


def main(query):
    answers = get_data.get_answers(f_path=args.f_path)

    # функция предобработки (нужна, если еще нет матрицы bert)
    # corp_prep = preprocess.corp_preprocessed(answers)


    query_prep = preprocess.query_preprocessed(query)  # предобработка запроса

    bert_matrix = indexation_bert.get_bert_matrix(bert_matrix_path=args.bert_matrix_path)  # bert матрица
    query_vec_bert = indexation_bert.query_vec_bert_indexation(bert_tokenizer, bert_model, query_prep)
    simularity_vec = simularity_bert.simularity_bert(bert_matrix, query_vec_bert) # косинусная близость

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
    argparser.add_argument("bert_matrix_path", help="Matrix saved path")
    # запрос
    argparser.add_argument("query", help="Query input")
    args = argparser.parse_args()
    main(query=args.query)



