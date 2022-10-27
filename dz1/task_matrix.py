def task_matrix(matrix, vectorizer):  # задание для матрицы
    matrix_freq = np.asarray(matrix.sum(axis=0)).ravel()  # матрица с частотами слов
    final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])  # матрица слова-их частоты

    # Самое частотное слово
    max_freq = np.max(matrix_freq)  # максимальная частота
    max_freq_ind = np.where(final_matrix[1] == str(max_freq))  # ее индекс в final_matrix
    max_freq_word = final_matrix[0][max_freq_ind]  # слово для этого индекса
    print('Самое частотное слово:', max_freq_word, '\n')

    # Самое нечастотное слово
    min_freq = np.min(matrix_freq)  # минимальная частота
    min_freq_ind = np.where(final_matrix[1] == str(min_freq)) # ее индекс
    min_freq_word = final_matrix[0][min_freq_ind] # слово для этого индекса
    print('Самые редкие слова', min_freq_word, '\n')

    # Слова, встречающиеся во всех документах
    words_in_all_docs_ind = index_matrix.toarray().all(axis=0)
    print('Слова, встречающиеся во всех документах', final_matrix[0][words_in_all_docs_ind], '\n')

    # Герой, упоминаемый чаще всего

    # моника
    monica_ind = (final_matrix[0] == 'мон') | (final_matrix[0] == 'моника')
    monica_freq = final_matrix[1][monica_ind]
    monica_freq = list(map(int, monica_freq))
    monica_freq = sum(monica_freq)  # сумма частот всех вариаций

    # рейчел
    rach_ind = (final_matrix[0] == 'рэйчел') | (final_matrix[0] == 'рейч')
    rach_freq = final_matrix[1][rach_ind]
    rach_freq = list(map(int, rach_freq))
    rach_freq = sum(rach_freq)

    # чендлер
    chan_ind = (final_matrix[0] == 'чендлер') | (final_matrix[0] == 'чэндлер') | (final_matrix[0] == 'чен')
    chan_freq = final_matrix[1][chan_ind]
    chan_freq = list(map(int, chan_freq))
    chan_freq = sum(chan_freq)

    # фиби
    phoebe_ind = (final_matrix[0] == 'фиби') | (final_matrix[0] == 'фибс')
    phoebe_freq = final_matrix[1][phoebe_ind]
    phoebe_freq = list(map(int, phoebe_freq))
    phoebe_freq = sum(phoebe_freq)

    # росс
    ross_ind = final_matrix[0] == 'росс'
    ross_freq = final_matrix[1][ross_ind]
    ross_freq = list(map(int, ross_freq))
    ross_freq = sum(ross_freq)

    # джоуи
    joey_ind = (final_matrix[0] == 'джоуи') | (final_matrix[0] == 'джо')
    joey_freq = final_matrix[1][joey_ind]
    joey_freq = list(map(int, joey_freq))
    joey_freq = sum(joey_freq)

    friends_freq = {'Моника': monica_freq,
                        'Рэйчел': rach_freq,
                        'Чендлер': chan_freq,
                        'Фиби': phoebe_freq,
                        'Росс': ross_freq,
                        'Джоуи': joey_freq}

    print('Самый популярный',
          max(friends_freq, key=friends_freq.get), '\n')