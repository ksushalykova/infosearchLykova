def task_dictionary(dict):  # задание для словаря

    # Самое частотное слово
    max_len = max(len(value) for value in dict.values())  # слово с самым длинным набором значений - ключей
    for key, value in dict.items():  # ключ этого слова
        if len(value) == max_len:
            word = key
            print('Самое частотное слово', word, '\n')

    # Самые нечастотные слова
    least_frequent_words = []
    min_len = min(len(value) for value in dict.values())  # слово с самым длинным набором значений - ключей
    for key, value in dict.items():
        if len(value) == min_len:
            least_frequent_words.append(key)
    print('Самые редкие слова', least_frequent_words, '\n')



    # Герой, упоминаемый чаще всего

    friends_freq = {'Моника': len(dict['моника'] + dict['мон']),
                        'Рэйчел': len(dict['рэйчел'] + dict['рейч']),
                        'Чендлер': len(dict['чендлер'] + dict['чэндлер'] + dict['чен']),
                        'Фиби': len(dict['фиби'] + dict['фибс']),
                        'Росс': len(dict['росс']),
                        'Джоуи': len(dict['джоуи'] + dict['джо'] + dict['джои'])}

    print('Самый популярный',
          max(friends_freq, key=friends_freq.get), '\n')


