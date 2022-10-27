import json
import jsonlines

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
