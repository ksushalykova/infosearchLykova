import argparse

import get_data
import preprocess
import indexation
import task_matrix
import task_dictionary

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # путь до корпуса и формат индексации читаются как аргументы, затем загружается и предобрабатывается корпус

    argparser.add_argument("f_path", help="Data directory")
    argparser.add_argument("matr_or_dict", help="Index format (dictionary or matrix)")
    args = argparser.parse_args()
    corpus = get_data.get_texts(f_path=args.f_path)
    corp_prep = preprocess.corp_preprocessed(corpus)

# в качестве аргументов нужно ввести путь до файла с корпусом и выбрать формат индексации (только matrix или dictionary)
# затем создается обратный индекс в виде матрицы или в виде словаря и с его помощью делается задание для обоих случаев

    if args.index_type == "matrix":
        matrix, vectorizer = indexation.matrix_indexation(corp_prep)
        task_matrix.task_matrix(matrix, vectorizer)

    if args.matr_or_dict == "dictionary":
        index_dict = indexation.dictionary_indexation(corp_prep)
        task_dictionary.task_dictionary(index_dict)
