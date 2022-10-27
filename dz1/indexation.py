from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize

# индексация словаря

def dictionary_indexation(corp_prep):
    dict = {}
    numerated_texts = []     # список с номерами текстов
    for i in range(len(corp_prep)):
        numerated_texts.append(i)

    for text_i, text_j in zip(numerated_texts, corp_prep):
        text_i = word_tokenize(text_i)
        for word in text_i:
            if word in dict.keys():            # если слово уже есть в ключах
                dict[word].append(text_j)
            else:                              # если слова еще нет в ключах
                dict[word] = [text_j]
    return dict


# индексация матрицы
# сохраняем векторайзер

def matrix_indexation(corp_prep):
    vectorizer = CountVectorizer(analyzer='word')
    matrix = vectorizer.fit_transform(corp_prep)
    return matrix, vectorizer