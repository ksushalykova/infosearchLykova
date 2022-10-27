import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

import pymorphy2

#morph = pymorphy2.MorphAnalyzer()

sw = stopwords.words('russian')

punctuation = '[\'\"!"#$%&()*\+,-\./:;<=>?@\[\]^_`{|}~„“«»*\—/\-‘’…–\.+☺❤๏̯͡๏ツتبالك\_]'

# загрузка MorphAnalyzer

def load_morph():
    morph = pymorphy2.MorphAnalyzer()
    return morph

# функция препроцессинга текста

def preprocessing(text, morph):
    clean_words = [re.sub('[A-Za-z]', '', w) for w in word_tokenize(text)]
    clean_words = [re.sub('[0-9]', '', w) for w in clean_words]
    clean_words = [w.strip(punctuation) for w in clean_words]
    clean_words = [re.sub('[\.-\/-\-]', ' ', w) for w in clean_words]
    clean_words = [re.sub('о_о', '', w) for w in clean_words]
    clean_words = [w.lower() for w in clean_words if w != '']
    clean_words = [w for w in clean_words if w not in sw]
    clean_words = [morph.parse(w)[0].normal_form for w in clean_words if w]
    clean_words = [w for w in clean_words if w not in sw]

    clean_words = str(clean_words)
    clean_words = re.sub('[,\[\]\']', '', clean_words)

    return clean_words


# функция получения корпуса-словаря с предобработанными ответами

def corp_preprocessed(answers):
    corp_prep = []
    morph = load_morph()
    for i in range(len(answers)):
        answer = preprocessing(text = answers[i], morph = morph)
        corp_prep.append(answer)

    return corp_prep


# функция предобработки запроса

def query_preprocessed(query):
    morph = load_morph()
    query_prep = preprocessing(text = query, morph = morph)

    return query_prep
